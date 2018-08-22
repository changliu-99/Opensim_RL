# Implement Chang_RL use tensorforce
import numpy as np
import sys

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time
import json

import numpy as np

from tensorforce import TensorForceError
from tensorforce.agents import Agent

# This was necessary for bazel, test if can be removed
logger = logging.getLogger(__name__)

from tensorforce.contrib.deepmind_lab import DeepMindLab
from tensorforce.execution import Runner

from osim.env import *
from osim.http.client import Client

import opensim

from MyModule import DDPGAgent_Chang
from MyModule import ProstheticsEnv_Chang
from MyModule import LayerNorm
import tensorflow as tf
import pandas as pd
import os

current_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_path,"action.csv")
print(path)
label=['abd_l','abd_r','add_l','add_r','bifemsh_l','bifemsh_r','gastroc_l',
    'glut_max_l','glut_max_r','hamstrings_l','hamstrings_r','iliopsoas_l',
    'iliopsoas_r','rect_fem_l','rect_fem_r','soleus_l','tib_ant_l','vasti_l',
    'vasti_r'];
actionData=pd.read_csv(path,names=label,header=0)
# rect_fem_l = actionData['rect_fem_l'].tolist()
# soleus_l = actionData['soleus_l'].tolist()
# tib_ant_l = actionData['tib_ant_l'].tolist()
# hamstrings_l = actionData['hamstrings_l'].tolist()
af = actionData.fillna(0)
a = af.values.tolist()

### add another file
current_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_path,"action_new.csv")
actionData_new=pd.read_csv(path,header=1)
label_new = ['bifemsh_l','gastroc_l','gastrocM_l','glut_max1_l','glut_max2_l',
    'glut_max3_l','glmed1','glmed2','glmed3','rect_fem_l','semimem_l','semiten_','soleus_r',
    'tibant_l','vaslat','vasmed_l']
af_new = actionData_new.fillna(0)
a_new = af_new.values.tolist()

# env = ProstheticsEnv(visualize=True)
# observation = env.reset()

# from MyModule import *
# check if use gpu
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# from MyModule import *

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=50000, type=int) #should be 100000
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
parser.add_argument('--resume', dest='resume', action='store_true', default=False)
args = parser.parse_args()



runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )
