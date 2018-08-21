# Derived from keras-rl
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Dropout,Convolution2D
from keras.optimizers import Adam

import numpy as np
import json
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math
import opensim

from MyModule import DDPGAgent_Chang
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


env = ProstheticsEnv(visualize=True)
observation = env.reset()

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

# Load walking environment

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

def process_observation(obs): #attempt to correct for the pelvis position
    # print(obs)
    for i in obs['body_pos']:
        if i != 'pelvis':
            # obs['body_pos'][i][0] = obs['body_pos'][i][0] - obs['body_pos']['pelvis'][0]
            obs['body_pos'][i][2] = obs['body_pos'][i][2] - obs['body_pos']['pelvis'][2]
    # obs['joint_pos']['back'][0] = obs['joint_pos']['back'][0] - obs['joint_pos']['ground_pelvis'][0]
    # obs['joint_pos']['back'][0] = obs['joint_pos']['back'][0] - obs['joint_pos']['ground_pelvis'][0]
    return obs

# wrap agent and critic
def actor_model(num_action,observation_shape):
    actor = Sequential()
#    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    # actor.add(Lambda())
    # actor.add(Convolution2D(64, 4, 4, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    actor.add(Flatten(input_shape=(1,) + observation_shape))
    actor.add(Dense(64))
    actor.add(Activation('relu'))
    # actor.add(Dropout(0.25)) #add by Chang
    # actor.add(Dense(32))
    # actor.add(Activation('selu'))
    # actor.add(Dropout(0.25)) #add by Chang
    # actor.add(Dense(128))
    # actor.add(Activation('relu'))
    actor.add(Dense(64))
    actor.add(Activation('relu'))
    actor.add(Dense(num_action))
    actor.add(Activation('tanh'))
    print(actor.summary())
    return actor
##
def critic_model(num_action,observation_shape):
    action_input = Input(shape=(num_action,), name='action_input')
    #observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    observation_input = Input(shape=(1,) + observation_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(128)(x)
    x = Activation('selu')(x)
    # x = Dropout(0.25)(x) #add by Chang
    # x = Dense(64)(x)
    # x = Activation('selu')(x)
    # x = Dropout(0.25)(x) #add by Chang
    x = Dense(64)(x)
    x = Activation('selu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())
    return critic, action_input
# Set up the agent for training
def build_agent(num_action,observation_shape):
    memory = SequentialMemory(limit=1000000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.get_action_space_size())
    actor = actor_model(num_action,observation_shape)
    critic,critic_action_input = critic_model(num_action,observation_shape)
    agent = DDPGAgent_Chang(nb_actions=num_action, actor=actor, critic=critic, critic_action_input=critic_action_input,
                  memory=memory, memory_interval=1,nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                  batch_size = 64,random_process=random_process, gamma=.995, target_model_update=1e-3,
                  delta_clip=1.)

    return agent
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# def experimentData_train():

#function to convert state_desc to list (https://www.endtoend.ai/blog/ai-for-prosthetics-3/)
def dict_to_list(state_desc):
    res = []

    # Body Observations
    for info_type in ['body_pos', 'body_pos_rot',
                      'body_vel', 'body_vel_rot',
                      'body_acc', 'body_acc_rot']:
        for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                          'femur_l', 'femur_r', 'head', 'pelvis',
                          'torso', 'pros_foot_r', 'pros_tibia_r']:
            res += state_desc[info_type][body_part]

    # Joint Observations
    # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
    for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
        for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                      'hip_l', 'hip_r', 'knee_l', 'knee_r']:
            res += state_desc[info_type][joint]

    # Muscle Observations
    for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                   'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                   'glut_max_l', 'glut_max_r',
                   'hamstrings_l', 'hamstrings_r',
                   'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                   'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
        res.append(state_desc['muscles'][muscle]['activation'])
        res.append(state_desc['muscles'][muscle]['fiber_force'])
        res.append(state_desc['muscles'][muscle]['fiber_length'])
        res.append(state_desc['muscles'][muscle]['fiber_velocity'])

    # Force Observations
    # Neglecting forces corresponding to muscles as they are redundant with
    # `fiber_forces` in muscles dictionaries
    for force in ['AnkleLimit_l', 'AnkleLimit_r',
                  'HipAddLimit_l', 'HipAddLimit_r',
                  'HipLimit_l', 'HipLimit_r', 'KneeLimit_l', 'KneeLimit_r']:
        res += state_desc['forces'][force]

        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']
    print (len(res))
    return res
def dict_to_list_Chang(state_desc):
    res = []
    pelvis = None
    for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
        if body_part in ["toes_r","talus_r"]:
            res += [0] * 9
            # print(body_part,len(res))
            continue
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
        else:
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
            res += cur

        # print(body_part,len(res)) #length = 62
    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]
        # print(joint,len(res)) # length = 95

    # for muscle in sorted(state_desc["muscles"].keys()):
    #     res += [state_desc["muscles"][muscle]["activation"]]
    #     res += [state_desc["muscles"][muscle]["fiber_length"]]
    #     res += [state_desc["muscles"][muscle]["fiber_velocity"]]
    #     # print(muscle)
    # cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    # res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    return res
def muscleActivationHack(act_l):
    # act_l should be an array
    # left_muscleIndex = np.array([1,3,5,7,8, 10, 12, 14, 16, 17, 18])
    # left_muscleIndex = left_muscleIndex-1
    # right_muscleIndex = np.array([2,4,6,9,11,13,15])
    # right_muscleIndex = right_muscleIndex-1

    leftActivation = np.where(act_l[left_muscleIndex]>0.5)
    for i in leftActivation:
        if i in np.array([0,2,4,6]):
            act_l[i+1] *= 0.5
    return act_l

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

env = ProstheticsEnv_Chang(args.visualize)
observation = env.reset(project = False) #keep as dictionary format
# print(observation)
nb_actions = env.action_space.shape[0]
observation_shape = env.observation_space.shape
# print(env.observation_space)
# print (observation_shape)
agent = build_agent(nb_actions,observation_shape)
# Total number of steps in training
nallsteps = args.steps
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

if args.train:
    # probably doesn't work this way
    if args.resume:
        agent.load_weights(args.model)
        print('resume')

    print('training')
    # agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.time_limit, log_interval=10000)
    # After training is done, we save the final weights.

    # implement my own fit
    nb_max_episode_steps = env.time_limit
    nb_max_start_steps = 0
    log_interval=10000
    max_steps = nallsteps
    visualize = False
    total_reward = 0
    done = False

    episode = np.int16(0)
    agent.step = np.int16(0)
    observation = None
    episode_reward = None
    episode_step = None

    head_pos = []
    head_pos_new = []
    action_repetition = 1
    print (agent.training)
    agent.training = True
    try:
        while agent.step < max_steps:
            if observation is None:  # start of a new episode
                # callbacks.on_episode_begin(episode)
                episode_step = np.int16(0)
                episode_reward = np.float32(0)

                observation = env.reset()
                # to start new simulations
                nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                # action = env.action_space.sample()
                # action_copy = action
                action = initialSample_action(a)
                # add initialize parameters for the models

                observation, reward, done, info = env.step(action)
                # observation = process_observation(observation)
                # project to np.array
                # observation = dict_to_list_Chang(observation)
                # print(env.real_reward())
                # print(env.reward())

        # print(observation)
            assert episode_reward is not None
            assert episode_step is not None
            assert observation is not None
            # print('initialization')
            # This is were all of the work happens. We first perceive and compute the action
                    # (forward step) and then use the reward to improve (backward step).
            v = np.array(observation).reshape((env.observation_space.shape[0]))
            action = agent.forward(v)

            reward = np.float32(0)
            accumulated_info = {}
            done = False
            abort = False

            observation, reward, done, info = env.step(action.tolist())

            # observation = process_observation(observation)
            # observation = dict_to_list_Chang(observation)



            # v = np.array(observation).reshape((env.observation_space.shape[0]))
            for key, value in info.items():
                if not np.isreal(value):
                    continue
                if key not in accumulated_info:
                    accumulated_info[key] = np.zeros_like(value)
                accumulated_info[key] += value

            # if done:
            #     break
            if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                done = True
            # print(agent.training)
            # print (agent.metrics_names)
            metrics = agent.backward(reward, terminal=done)
            # print (metrics)
            episode_reward += reward

            step_logs = {
                'action': action,
                'observation': observation,
                'reward': reward,
                'metrics': metrics,
                'episode': episode,
                'info': accumulated_info,
            }

            episode_step += 1
            agent.step += 1
            # print(env.reward(),'/',max_steps)
            if done:
                # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # perform one more forward-backward call and simply ignore the action before
                # resetting the environment. We need to pass in `terminal=False` here since
                # the *next* state, that is the state of the newly reset environment, is
                # always non-terminal by convention.

                agent.forward(v)
                agent.backward(0., terminal=False)
                episode_logs = {
                'episode_reward': episode_reward,
                'nb_episode_steps': episode_step,
                'nb_steps': agent.step,
                }
                print(episode_reward, ' steps=',episode_step,' ',agent.step,'/',max_steps)
                episode += 1
                observation = None
                episode_step = None
                episode_reward = None


    except KeyboardInterrupt:
        did_abort = True
        agent.save_weights(args.model, overwrite=True)
    log_filename = '/Users/liuchang/dqn_test_log.json'

    with open(log_filename, "w") as write_file:
        json.dump(head_pos, write_file)
        json.dump(head_pos_new,write_file)
    agent.save_weights(args.model, overwrite=True)

# If TEST and TOKEN, submit to csrowdAI
if args.test:
    visualize = True
    print('test')
    agent.load_weights(args.model)
    # Settings

    total_reward = 0
    total_real_reward = 0
    # Create environment
    observation = env.reset()
    # print(observation)
    # project_observation = dict_to_list_Chang(observation)
    # print(project_observation)
    agent.test(env, nb_episodes=3, visualize=True, nb_max_episode_steps=1000)

    # Run a single step
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    # agent.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=500)

    # for i in range(1000):
    #     v = np.array(observation).reshape((env.observation_space.shape[0]))
    #     action = agent.forward(v)
    #     [observation, reward, done, info] = env.step(action.tolist())
    #     # observation = process_observation(observation)
    #     # project to np.array
    #     # project_observation = dict_to_list_Chang(observation)
    #
    #     real_reward = env.real_reward()
    #     total_reward += reward
    #     total_real_reward += real_reward
    #
    #     if observation[0] < 0.6:
    #         break
    # print(total_reward)
    # print(total_real_reward)

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
