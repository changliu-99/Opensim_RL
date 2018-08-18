from osim.env import ProstheticsEnv
from osim.env import OsimModel
import pandas as pd
import numpy as np
import random
# read csv
path = '/Users/liuchang/Google Drive/2018_Research/OpenSim_RL/Code/ExperimentData/action.csv'
label=['abd_l','abd_r','add_l','add_r','bifemsh_l','bifemsh_r','gastroc_l',
    'glut_max_l','glut_max_r','hamstrings_l','hamstrings_r','iliopsoas_l',
    'iliopsoas_r','rect_fem_l','rect_fem_r','soleus_l','tib_ant_l','vasti_l',
    'vasti_r'];
actionData=pd.read_csv(path,names=label,header=0)
abd_l = actionData['abd_l'].tolist()
af = actionData.fillna(0)
a = af.values.tolist()
print(len(a))
print(a)

env = ProstheticsEnv(visualize=True)
observation = env.reset()

# class FixedActionAgent(Agent):
#     """
#     An agent that choose one fixed action at every timestep.
#     """
#     def __init__(self, env):
#         self.action = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
#
#     def act(self, observation):
#         return self.action
def initialSample_action(action,experiment_act):
    # c = list(range(0, 256))
    ind = np.asscalar(np.random.choice(len(experiment_act),1))
    print(ind)
    action[9] = experiment_act[ind][9]
    action[13] = experiment_act[ind][13]
    action[15] = experiment_act[ind][15]
    action[16] = experiment_act[ind][16]
    # print(action)
    return action

for i in range(100):
    action = env.action_space.sample()
    action_copy = action
    action = initialSample_action(action_copy,a)
    # print(action)
# step_size = 0.01
# for i in range(100):
#     # if mod(i)==0:
#     j = 0
#     if np.mod(i,2) == 0:
#         action = a[j]
#         j += 1
#         observation, reward, done, info = env.step(action*2, project = False)
#     else:
#         action = a[j]
#         observation, reward, done, info = env.step(action*2, project = False)
#
#     # print(observation['body_pos']['pelvis'])
#     # print(env.get_state_desc()['body_pos']['pelvis'])
#
#
#     # for keys in env.get_state_desc()['body_pos'].keys():
#     #     print(keys)
#     # print(env.action_space.tolist())
# total_reward = 0.0
