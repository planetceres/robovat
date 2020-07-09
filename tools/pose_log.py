"""Logging object poses.
"""


import random
import string
import mlflow

from robovat.math import Pose 
from robovat.simulation.body import Body


def random_string(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def log_pose():
    experiment_name = random_string()
    #mlflow.set_tracking_uri()
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.start_run()
    #run = mlflow.tracking.MlflowClient.create_run(experiment_id)
    #run_id = run.info.run_id
 


def helper(info, objs, action):
    #experiment = mlflow.get_experiment_by_name('random')
    #experiment_id = experiment.experiment_id

    mlflow.log_metric("start_X" , action[0], step = info)
    mlflow.log_metric("start_Y" , action[1], step = info)
    mlflow.log_metric("motion_X" , action[2], step = info)
    mlflow.log_metric("motion_Y" , action[3], step = info)
    for body in objs:

        mlflow.log_metric(body.name+"_X" , body.pose.x, step = info)
        mlflow.log_metric(body.name+"_Y" , body.pose.y, step = info)
        mlflow.log_metric(body.name+"_Z" , body.pose.z, step = info)
        mlflow.log_metric(body.name+"_roll" , body.pose.euler[0], step = info)
        mlflow.log_metric(body.name+"_pitch" , body.pose.euler[1], step = info)
        mlflow.log_metric(body.name+"_yaw" , body.pose.euler[2], step = info)



        '''batch = {}
        batch.update( {body.name+"_X" : body.pose.x} )
        batch.update( {body.name+"_Y" : body.pose.y} )
        batch.update( {body.name+"_Z" : body.pose.z} )
        batch.update( {body.name+"_roll" : body.pose.euler[0]} )
        batch.update( {body.name+"_pitch" : body.pose.euler[1]} )
        batch.update( {body.name+"_yaw" : body.pose.euler[2]} )

        mlflow.log_metrics(metrics = batch)'''






