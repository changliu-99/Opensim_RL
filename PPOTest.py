import numpy as np
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Dropout,Convolution2D
from keras.optimizers import Adam

from osim.env import *
from osim.http.client import Client
import argparse
import math
import opensim

from MyModule import DDPGAgent_Chang
import tensorflow as tf
import pandas as pd
import os

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Create an OpenAIgym environment
env = ProstheticsEnv_Chang(visualize = True)
# observation = env.reset(project = False) #keep as dictionary format

# Network as list of layers
network_spec = [
    dict(type='dense', size=64, activation='relu'),
    dict(type='dense', size=32, activation='tanh')
]

agent = PPOAgent(
    states=env.get_observation(),
    actions=env.get_activation(),
    network=network_spec,
    batch_size=4096,
    # BatchAgent
    keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=1,
    # DistributionModel
    distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    summary_spec=None,
    distributed_spec=None
)

# Create the runner
runner = Runner(agent=agent, environment=env)
