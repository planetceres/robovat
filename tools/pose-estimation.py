'''
estimates next pose after a given action is performed
'''

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
    '''
    returns datasets consisting of initial-pose, final-pose and action
    from log files created using log_pose.py
    '''
    list = os.listdir(path) 
    num_files = len(list)
    num_objects = int((num_files-4)/6)

    '''extract action components from log files'''

    a1 = np.genfromtxt(path+ 'start_X', delimiter=' ')[:, [1]]
    a2 = np.genfromtxt(path+ 'start_Y', delimiter=' ')[:, [1]]
    a3 = np.genfromtxt(path+ 'motion_X', delimiter=' ')[:, [1]]
    a4 = np.genfromtxt(path+ 'motion_Y', delimiter=' ')[:, [1]]
    action1 = np.concatenate((a1,a2,a3,a4), axis = 1)
    action1 = np.delete(action1, 0, axis=0) #delete the first column since there is no corresponding initial pose
    
    '''duplicate action column to account for number of objects'''

    tpl = ()
    for i in range(0, num_objects):
        tpl = tpl + (action1, )
    action = np.concatenate(tpl, axis = 0)


    '''extract pose data from log files and create initial and final pose datasets'''

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
       
    return (initial_pose, final_pose, action)


def build_model(train_dataset, isyaw):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
      ])
    if isyaw:
        r = 0.00001
    else:
       r = 0.001
    optimizer = tf.keras.optimizers.RMSprop(r)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def predict_for(unknown, dataset, test_dataset = None):
    '''
    creates a model to estimate the values of the unknown final pose component using the tf model above 
    returns the value and the ground truth
    '''

    components = ['X2', 'Y2', 'Z2', 'roll2', 'pitch2', 'yaw2']
    
   
    if test_dataset is not None:
        train_dataset = dataset
    else:
        '''separate dataset into test and train data'''
        train_dataset = dataset.sample(frac=0.8,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

    '''separate train labels and test labels, and drop the rest of final pose data.'''
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
   
    ''' normalize training and test data '''
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)


    model = build_model(train_dataset, unknown == 'yaw2')

    '''train for 1000 epochs with early-stopping'''

    
    if unknown == 'Yaw2':
        EPOCHS = 2000
        pat = 400
    else:
        EPOCHS = 1000 
        pat = 200



    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

    early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    '''Visualize the model's training progress'''

    #ehist = pd.DataFrame(early_history.history)
    #ehist['epoch'] = early_history.epoch
    #print(ehis.tail())

    '''plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': early_history}, metric = "mean_absolute_error")
    plt.ylim([0, 0.3])
    plt.ylabel('MAE ['+unknown+']')
    plt.show()

    plotter.plot({'Basic': early_history}, metric = "mean_squared_error")
    plt.ylim([0, 0.25])
    plt.ylabel('MSE ['+unknown+']')
    plt.show()
              '''

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} {}".format(mae, unknown))


    test_predictions = model.predict(normed_test_data).flatten()

    '''Visualize the test predictions and prediction error'''

    ''''a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values ['+unknown+']')
    plt.ylabel('Predictions ['+unknown+']')
    lims = [-1, 2]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)   
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error"+unknown)
    _ = plt.ylabel("Count")
    plt.show()'''

    return test_predictions, test_labels

def find_forward_error(path, step-pose1, step-pose2, step-action):
    #TODO: make sure you delete all previous experiments using 'mlflow gc' 
    list = os.listdir(path) 
    num_files = len(list)  
   
    initial_pose = np.empty((0,6))
    final_pose = np.empty((0,6))
    action = np.empty((0,4))  

    '''create initial pose, final pose and action datasets for all Runs'''

    for i in range(0, num_files):

        p = Path(path)
        subdirectories = [x for x in p.iterdir() if x.is_dir()]
        for sub in subdirectories:
            p, q, r = get_data(path+str(i)+'/sub/metrics/')
            initial_pose = np.concatenate((initial_pose, p), axis=0)
            final_pose = np.concatenate((final_pose, q), axis=0)
            action = np.concatenate((action, r), axis=0)


    dataset = pd.DataFrame({'X1': initial_pose[:, 0], 'Y1': initial_pose[:, 1], 'Z1': initial_pose[:, 2], 'roll1': initial_pose[:, 3], 'pitch1': initial_pose[:, 4], 'yaw1': initial_pose[:, 5],
                            'a1': action[:, 0], 'a2': action[:, 1], 'a3': action[:, 2], 'a4': action[:, 3], 'X2': final_pose[:, 0], 'Y2': final_pose[:, 1], 'Z2': final_pose[:, 2], 
                            'roll2': final_pose[:, 3], 'pitch2': final_pose[:, 4], 'yaw2': final_pose[:, 5]})

    dataset2 = pd.DataFrame({'X1': step-pose1[:, 0], 'Y1': step-pose1[:, 1], 'Z1': step-pose1[:, 2], 'roll1': step-pose1[:, 3], 'pitch1': step-pose1[:, 4], 'yaw1': step-pose1[:, 5],
                            'a1': step-action[:, 0], 'a2': step-action[:, 1], 'a3': step-action[:, 2], 'a4': step-action[:, 3], 'X2': step-pose2[:, 0], 'Y2': step-pose2[:, 1], 'Z2': step-pose2[:, 2], 
                            'roll2': step-pose2[:, 3], 'pitch2': step-pose2[:, 4], 'yaw2': step-pose2[:, 5]})



    '''Create models to estimate each final pose component using the predict_for function'''
    #TODO: From this point onwards figure out how to account for each pose component error in each movable and rewrite...

    X, predicted_X = predict_for('X2',dataset, dataset2)
    Y, predicted_Y = predict_for('Y2',dataset, dataset2)
    Z, predicted_Z = predict_for('Z2',dataset, dataset2)
    roll, predicted_roll = predict_for('roll2',dataset, dataset2)
    pitch, predicted_pitch = predict_for('pitch2',dataset, dataset2)
    yaw, predicted_yaw = predict_for('yaw2',dataset, dataset2)

    '''Save individual data files of each final pose component estimate vs ground truth'''

    X_result= np.column_stack((X, predicted_X))
    print(str(abs(X - predicted_X)))
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


def main():


    initial_pose = np.empty((0,6))
    final_pose = np.empty((0,6))
    action = np.empty((0,4))

    '''create initial pose, final pose and action datasets for all Runs'''

    for i in range(1, 28):
        p, q, r = get_data('curis-project/robovat-curis-mona/docs/logs/Run'+str(i)+'/')
        initial_pose = np.concatenate((initial_pose, p), axis=0)
        final_pose = np.concatenate((final_pose, q), axis=0)
        action = np.concatenate((action, r), axis=0)


    dataset = pd.DataFrame({'X1': initial_pose[:, 0], 'Y1': initial_pose[:, 1], 'Z1': initial_pose[:, 2], 'roll1': initial_pose[:, 3], 'pitch1': initial_pose[:, 4], 'yaw1': initial_pose[:, 5],
                            'a1': action[:, 0], 'a2': action[:, 1], 'a3': action[:, 2], 'a4': action[:, 3], 'X2': final_pose[:, 0], 'Y2': final_pose[:, 1], 'Z2': final_pose[:, 2], 
                            'roll2': final_pose[:, 3], 'pitch2': final_pose[:, 4], 'yaw2': final_pose[:, 5]})

    '''Create models to estimate each final pose component using the predict_for function'''

    X, predicted_X = predict_for('X2',dataset)
    Y, predicted_Y = predict_for('Y2',dataset)
    Z, predicted_Z = predict_for('Z2',dataset)
    roll, predicted_roll = predict_for('roll2',dataset)
    pitch, predicted_pitch = predict_for('pitch2',dataset)
    yaw, predicted_yaw = predict_for('yaw2',dataset)

    '''Save individual data files of each final pose component estimate vs ground truth'''

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














