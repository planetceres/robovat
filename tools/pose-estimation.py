'''
estimates next pose after a given action is performed
'''
# Use seaborn for pairplot
#!pip install -q seaborn

# Use some functions from tensorflow_docs
#!pip install -q git+https://github.com/tensorflow/docs

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling



def get_data(path):

    list = os.listdir(path) 
    num_files = len(list)
    print (num_files)

    num_objects = int((num_files-4)/6)

    a1 = np.genfromtxt(path+ 'start_X', delimiter=' ')[:, [1]]
    a2 = np.genfromtxt(path+ 'start_Y', delimiter=' ')[:, [1]]
    a3 = np.genfromtxt(path+ 'motion_X', delimiter=' ')[:, [1]]
    a4 = np.genfromtxt(path+ 'motion_Y', delimiter=' ')[:, [1]]

    action1 = np.concatenate((a1,a2,a3,a4), axis = 1)
    action1 = np.delete(action1, 0, axis=0)
    tpl = ()

    for i in range(0, num_objects):
        tpl = tpl + (action1, )

    action = np.concatenate(tpl, axis = 0)
    print(action)
    print(action.shape)

    initial_pose = np.empty((0, 6))
    final_pose = np.empty((0, 6))


    for i in range(0, num_objects):
        X1 = np.genfromtxt(path+'movable_'+str(i)+'_X', delimiter=' ')[:, [1]]
        Y1 = np.genfromtxt(path+'movable_'+str(i)+'_Y', delimiter=' ')[:, [1]]
        Z1 = np.genfromtxt(path+'movable_'+str(i)+'_Z', delimiter=' ')[:, [1]]
        roll1 = np.genfromtxt(path+'movable_'+str(i)+'_roll', delimiter=' ')[:, [1]]
        pitch1 = np.genfromtxt(path+'movable_'+str(i)+'_pitch', delimiter=' ')[:, [1]]
        yaw1 = np.genfromtxt(path+'movable_'+str(i)+'_yaw', delimiter=' ')[:, [1]]

        pose1 = np.concatenate((X1, Y1, Z1, roll1, pitch1, yaw1), axis=1)
        pose2 = np.delete(pose1, 0, axis=0)
        pose1 = np.delete(pose1, -1, axis=0)
        
        initial_pose = np.concatenate((initial_pose, pose1), axis=0)
        final_pose = np.concatenate((final_pose, pose2), axis=0)

    print(initial_pose.shape)
    print(final_pose.shape)
    print(action.shape)
       
    return (initial_pose, final_pose, action)


def build_model(train_dataset):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
      ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def predict_for(unknown, dataset):

    components = ['X2', 'Y2', 'Z2', 'roll2', 'pitch2', 'yaw2']
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    print(train_dataset.shape)
    print(test_dataset.shape)
    train_stats = train_dataset.describe()
    for component in components:
        train_stats.pop(component)
    train_stats = train_stats.transpose()
     
    for component in components:
        if component != unknown:
            train_dataset.pop(component)
            test_dataset.pop(component)

    train_labels = train_dataset.pop(unknown)
    test_labels = test_dataset.pop(unknown)
   

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    model = build_model(train_dataset)

    print(model.summary())
  
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)


    EPOCHS = 1000
    
    '''history = model.fit(normed_train_data, train_labels,
                        epochs=EPOCHS, validation_split = 0.2, verbose=0,
                        callbacks=[tfdocs.modeling.EpochDots()])



    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print('\n\n\n')
    print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mean_absolute_error")
    plt.ylim([0, 10])
    plt.ylabel('MAE ['+unknown+']')

    plt.show()'''

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

    early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    ehist = pd.DataFrame(early_history.history)
    ehist['epoch'] = early_history.epoch
    print('\n\n\n')
    print(ehist.tail())

    '''plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Early Stopping': early_history}, metric = "mean_absolute_error")
    plt.ylim([0, 10])
    plt.ylabel('MAE ['+unknown+']')
    plt.show()'''


    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} {}".format(mae, unknown))


    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values ['+unknown+']')
    plt.ylabel('Predictions ['+unknown+']')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error"+unknown)
    _ = plt.ylabel("Count")
    plt.show()

    return test_predictions, test_labels


def main():
    initial_pose = np.empty((0,6))
    final_pose = np.empty((0,6))
    action = np.empty((0,4))
    for i in range(1, 16):
        p, q, r = get_data('curis-project/robovat-curis-mona/docs/logs/Run'+str(i)+'/')
        print ('\n\n')
        print(p.shape)
        initial_pose = np.concatenate((initial_pose, p), axis=0)
        final_pose = np.concatenate((final_pose, q), axis=0)
        action = np.concatenate((action, r), axis=0)
        
    print(action.shape)
    print(final_pose.shape)
    print(initial_pose.shape)



    dataset = pd.DataFrame({'X1': initial_pose[:, 0], 'Y1': initial_pose[:, 1], 'Z1': initial_pose[:, 2], 'roll1': initial_pose[:, 3], 'pitch1': initial_pose[:, 4], 'yaw1': initial_pose[:, 5],
                            'a1': action[:, 0], 'a2': action[:, 1], 'a3': action[:, 2], 'a4': action[:, 3], 'X2': final_pose[:, 0], 'Y2': final_pose[:, 1], 'Z2': final_pose[:, 2], 
                            'roll2': final_pose[:, 3], 'pitch2': final_pose[:, 4], 'yaw2': final_pose[:, 5]})

    print(dataset.tail())
    
    print(dataset.isna().sum())
    


    #sns.pairplot(train_dataset[["X1", "a1", "X2"]], diag_kind="kde")
    #plt.show()

    X, predicted_X = predict_for('X2',dataset)
    Y, predicted_Y = predict_for('Y2',dataset)
    Z, predicted_Z = predict_for('Z2',dataset)
    roll, predicted_roll = predict_for('roll2',dataset)
    pitch, predicted_pitch = predict_for('pitch2',dataset)
    yaw, predicted_yaw = predict_for('yaw2',dataset)

    
    X_result= np.column_stack((X, predicted_X))
    np.savetxt("X-result.csv", X_result, delimiter=",")

    Y_result= np.column_stack((Y, predicted_Y))
    np.savetxt("Y-result.csv", Y_result, delimiter=",")

    Z_result= np.column_stack((Z, predicted_Z))
    np.savetxt("Z-result.csv", Z_result, delimiter=",")

    roll_result= np.column_stack((roll, predicted_roll))
    np.savetxt("roll-result.csv", roll_result, delimiter=",")

    pitch_result= np.column_stack((pitch, predicted_pitch))
    np.savetxt("pitch-result.csv", pitch_result, delimiter=",")

    yaw_result= np.column_stack((yaw, predicted_yaw))
    np.savetxt("yaw-result.csv", yaw_result, delimiter=",")
    


if __name__ == '__main__':
    main()














