# Derived from keras-rl
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization,Activation, Lambda,Flatten, Input, concatenate, Dropout,Convolution2D
from keras.optimizers import Adam

import numpy as np
import json
from rl.agents import DDPGAgent
# from rl.memory import SequentialMemory
from MyModule.memory import SequentialMemory
# from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math
import opensim

from MyModule import DDPGAgent_Chang_2
from MyModule import ProstheticsEnv_Chang
from MyModule import LayerNorm
from MyModule import OrnsteinUhlenbeckProcess
import tensorflow as tf
import pandas as pd
import os
##
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

#env = ProstheticsEnv(visualize=True)
#observation = env.reset()

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
memoryview# Load walking environment

# Create networks for DDPG
# Next, we build a very simple model.
step_size = 0.01
left_muscleIndex = np.array([1,3,5,7,8, 10, 12, 14, 16, 17, 18])
left_muscleIndex = left_muscleIndex-1
right_muscleIndex = np.array([2,4,6,9,11,13,15])
right_muscleIndex = right_muscleIndex-1

def initialSample_action(action,experiment_act):
    ind = random.sample(c = list(range(0, 256)))
    action[9,13,15,16] = experiment_act[ind][9,13,15,16]


# wrap agent and critic
def actor_model(num_action,observation_shape):
    actor = Sequential()
#    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    # actor.add(Lambda())
    # actor.add(Convolution2D(64, 4, 4, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    actor.add(Flatten(input_shape=(1,) + observation_shape))
    actor.add(Dense(64))
    actor.add(LayerNorm())
    actor.add(Activation('selu'))
    actor.add(Dense(64))
    # actor.add(BatchNormalization(axis=1,input_shape=64))
    actor.add(LayerNorm())
    actor.add(Activation('selu'))
    actor.add(Dense(num_action))
    # actor.add(BatchNormalization(axis=1,input_shape=64))
    # actor.add(LayerNorm())
    actor.add(Activation('tanh'))
    actor.add(Lambda(lambda x: x*0.5+0.5))
    print(actor.summary())
    print(actor.get_weights())
    return actor
##
def critic_model(num_action,observation_shape):
    action_input = Input(shape=(num_action,), name='action_input')
    #observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    observation_input = Input(shape=(1,) + observation_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(128)(x)
    x = LayerNorm()(x)
    x = Activation('selu')(x)
    x = Dense(64)(x)
    x = LayerNorm()(x)
    x = Activation('selu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())
    return critic, action_input
# Set up the agent for training
def build_agent(num_action,observation_shape):
    memory = SequentialMemory(limit=1000000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2,
    size=env.get_action_space_size(),sigma_min=0.05,n_steps_annealing=1000000)
    actor = actor_model(num_action,observation_shape)
    critic,critic_action_input = critic_model(num_action,observation_shape)
    agent = DDPGAgent_Chang_2(nb_actions=num_action, actor=actor, critic=critic, critic_action_input=critic_action_input,
                  memory=memory, memory_interval=1,nb_steps_warmup_critic=300, nb_steps_warmup_actor=300,
                  train_interval=1, batch_size = 100,random_process=random_process, gamma=.995, target_model_update=0.001,
                  delta_clip=1.,param_noise=True)

    return agent
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# def experimentData_train():

#function to convert state_desc to list (https://www.endtoend.ai/blog/ai-for-prosthetics-3/)

def initialSample_action(experiment_act):
    # c = list(range(0, 256))
    action = [0]*19
    ind = np.asscalar(np.random.choice(len(experiment_act),1))
    # print(ind)
    action[9] = experiment_act[ind][9]
    action[13] = experiment_act[ind][13]
    action[15] = experiment_act[ind][15]
    action[16] = experiment_act[ind][16]
    # print(action)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.get_action_space_size())
    action += random_process.sample()
    return action

def initialSample_action_new(experiment_act):
    # c = list(range(0, 256))
    action = [0]*19
    ind = np.asscalar(np.random.choice(len(experiment_act),1))
    # print(ind)
    action[0] = experiment_act[ind][6]
    action[4] = experiment_act[ind][0]
    action[6] = experiment_act[ind][1]
    action[7] = experiment_act[ind][3]
    action[16] = experiment_act[ind][13]
    action[17] = experiment_act[ind][14]
    action[13] = experiment_act[ind][9]
    # print(action)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.get_action_space_size())
    action += random_process.sample()
    return action

def injectNoise(action):
    random_process = OrnsteinUhlenbeckProcess(theta=.1, mu=0., sigma=.02, size=env.get_action_space_size())
    action += random_process.sample()
    return action

env = ProstheticsEnv_Chang(args.visualize,skip_frame=4)
obs = env.reset(project= False)
print(obs["body_pos"]["pros_foot_r"])

nb_actions = env.action_space.shape[0]
observation_shape = env.observation_space.shape
print(observation_shape)

agent = build_agent(nb_actions,observation_shape)
# Total number of steps in training
nallsteps = args.steps
agent.compile([Adam(lr=.0001, clipnorm=1.),Adam(lr=0.0003,clipnorm = 1)], metrics=['mae'])

if args.train:
    # TODO: warp this training as function
    if args.resume:
        agent.load_weights(args.model)
        print('resume')

    print('training')
    # print(init_action)
    agent.train(args.steps)
    # agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.time_limit, log_interval=10000)
    # After training is done, we save the final weights.

    # implement my own fit
    agent.save_weights(args.model, overwrite=True)
    print ('done training')
# If TEST and TOKEN, submit to csrowdAI
if args.test:
    visualize = True
    print('test')
    agent.load_weights(args.model)
    # Settings

    total_reward = 0
    total_real_reward = 0
    # Create environment
    env = ProstheticsEnv_Chang(args.visualize,skip_frame=1)

    # print(observation)
    observation = env.reset()
    # print(observation)
    # project_observation = dict_to_list_Chang(observation)
    # print(project_observation)
    # agent.test(env, nb_episodes=3, visualize=False, nb_max_episode_steps=1000)

    # Run a single step
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    # agent.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=500)
    agent.rollout = False
    agent.action_noise = False
    agent.random_process.reset_states()
    print(agent.random_process.current_sigma)
    for i in range(1000):
        v = np.array(observation).reshape((env.observation_space.shape[0]))
        action = agent.forward(v)
        [observation, reward, done, info] = env.step(action.tolist())
        # observation = process_observation(observation)
        # project to np.array
        # project_observation = dict_to_list_Chang(observation)
        # print(observation)
        real_reward = env.real_reward()
        total_reward += reward
        total_real_reward += real_reward

        if observation[0] < 0.6:
            break

    print(total_reward)
    print(total_real_reward)

        # if done:
        #     observation = env.reset()
            # if not observation:
            #     break

    # client.submit()

# If TEST and no TOKEN, run some test experiments
if args.token:
    agent.load_weights(args.model)
    remote_base = 'http://grader.crowdai.org:1729'
    client = Client(remote_base)

    # Create environment
    observation = client.env_create(args.token)

    # Run a single step
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    while True:
        v = np.array(observation).reshape((env.observation_space.shape[0]))
        action = agent.forward(v)
        [observation, reward, done, info] = client.env_step(action.tolist())
        observation = process_observation(observation)
        total_reward += reward
        if done:
            observation = client.env_reset()
            if not observation:
                break

    client.submit()
    # Finally, evaluate our algorithm for 1 episode.
    #
