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
 


def helper(info, objs):
    #experiment = mlflow.get_experiment_by_name('random')
    #experiment_id = experiment.experiment_id

    batch = {}

    for body in objs:
        batch.update( {body.name+"_X" : body.pose.x} )
        batch.update( {body.name+"_Y" : body.pose.y} )
        batch.update( {body.name+"_Z" : body.pose.z} )
        batch.update( {body.name+"_roll" : body.pose.euler[0]} )
        batch.update( {body.name+"_pitch" : body.pose.euler[1]} )
        batch.update( {body.name+"_yaw" : body.pose.euler[2]} )

        mlflow.log_metrics(metrics = batch)






