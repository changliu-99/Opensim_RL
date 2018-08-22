import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from osim.env import *
## ADD by Chang
class ProstheticsEnv_Chang(OsimEnv):
    # model_path = os.path.join(os.path.dirname(__file__), '..models/gait14dof22musc_pros_20180507.osim'
    prosthetic = True
    model = "3D"
    def get_model_key(self):
        return self.model + ("_pros" if self.prosthetic else "")

    time_limit = 300

    def __init__(self, visualize = True, integrator_accuracy = 5e-5*2,skip_frame=5):
        self.model_paths = {}
        self.model_paths["3D_pros"] = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_pros_20180507.osim')
        self.model_paths["3D"] = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_20170320.osim')
        self.model_paths["2D_pros"] = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_planar_pros_20180507.osim')
        self.model_paths["2D"] = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_planar_20170320.osim')
        self.model_path = self.model_paths[self.get_model_key()]
        super(ProstheticsEnv_Chang, self).__init__(visualize = visualize, integrator_accuracy = integrator_accuracy)
        self.skip_frame = skip_frame

    def change_model(self, model='3D', prosthetic=True, difficulty=0, seed=None):
        if (self.model, self.prosthetic) != (model, prosthetic):
            self.model, self.prosthetic = model, prosthetic
            self.load_model(self.model_paths[self.get_model_key()])

    def is_done(self):
        state_desc = self.get_state_desc()
        done = not (state_desc["body_pos"]["pelvis"][1] > 0.6  and
                    state_desc["body_pos_rot"]["pelvis"][2] < 1.5 and
                    state_desc["body_pos_rot"]["pelvis"][2] > -1.5)
        # print(state_desc["body_pos_rot"]["pelvis"][2])
        # print(done)
        return done

    ## Values in the observation vector
    # y, vx, vy, ax, ay, rz, vrz, arz of pelvis (8 values)
    # x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (9*6 values)
    # rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 values)
    # activation, fiber_len, fiber_vel for all muscles (3*18)
    # x, y, vx, vy, ax, ay ofg center of mass (6)
    # 8 + 9*6 + 8*3 + 3*18 + 6 = 146
    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        for body_part in ["pelvis", "head","torso","toes_l","pros_foot_r","talus_l","talus_r"]:
            # if self.prosthetic and body_part in ["toes_r","talus_r"]:
            #     res += [0] * 9
            #     continue
            # add information about the toe position
            if self.prosthetic and body_part in ["talus_r"]:
                res += [0] * 9
                continue
            if self.prosthetic and body_part in ["pros_foot_r"]:
                cur = []

                cur += state_desc["body_pos"][body_part][0:2]
                cur_upd = cur
                cur_upd[:2] = [cur[i] - state_desc["body_pos"]["pelvis"][i] for i in range(2)]
                res += cur_upd
                res += [0] * 7
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
                res += cur_upd

        for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        # for muscle in sorted(state_desc["muscles"].keys()):
        #     res += [state_desc["muscles"][muscle]["activation"]]
        #     res += [state_desc["muscles"][muscle]["fiber_length"]]
        #     res += [state_desc["muscles"][muscle]["fiber_velocity"]]
        #
        # cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        # res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res

    def step(self, action, project = True, skip_frame=5):
        action = np.clip(action, 0, 1)
        reward = 0
        info = {'original_reward':0}
        for _ in range(skip_frame):
            self.prev_state_desc = self.get_state_desc()
            self.osim_model.actuate(action)
            self.osim_model.integrate()

            if project:
                obs = self.get_observation()
            else:
                obs = self.get_state_desc()
            reward += self.reward(action)
            info['original_reward'] += self.real_reward()
            # print(info['original_reward'])
            if self.is_done():
                break
        return [ obs, reward, self.is_done(),info ]

    def step_begin(self, action, project = True):
        action = np.clip(action, 0, 1)
        reward = 0
        info = {'original_reward':0}

        self.prev_state_desc = self.get_state_desc()
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()
        reward += self.reward(action)
        info['original_reward'] += self.real_reward()
            # print(info['original_reward'])

        return [ obs, reward, self.is_done(),info ]

    def get_observation_space_size(self):
        if self.prosthetic == True:
            #give up all the muscle state and COM
            return 95
            # return 158
        return 167

    def reward(self,action):
        # state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return self.compute_reward(action)+self.real_reward()*0.5

    def compute_reward(self,action):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        # reward_hack = state_desc["body_pos"]["pelvis"][0] * 0.1 #to move in space
        reward_hack = 0
        reward_hack += 0.2  # small reward for being alive
        reward_hack -= 1e-3 * np.square(action).sum()
        # reward_hack += min(0, state_desc["body_pos"]["head"][0] - state_desc["body_pos"]["pelvis"][0]) * 0.2  # penalty for head behind pelvis
        # reward_hack -= sum([max(0.0, k - 0.1) for k in [self.state_desc[""], self.current_state[10]]]) * 0.02  # penalty for straight legs
        # reward_hack -= abs(state_desc["body_acc"]["pelvis"][0])*0.1
        # reward_hack += min(0,state_desc["body_pos"]["pelvis"][1]-0.8) #penalty for fall
        return reward_hack

    def real_reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0)**2
