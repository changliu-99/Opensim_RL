from osim.env import ProstheticsEnv
from osim.env import OsimModel
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

for i in range(100):
    observation, reward, done, info = env.step(env.action_space.sample(), project = False)
    print(observation['body_pos']['pelvis'])
    print(env.get_state_desc()['body_pos']['pelvis'])
    env.get_state_desc()['body_pos']['pelvis'][0] = 0
    print(env.get_state_desc()['body_pos']['pelvis'])
    # for keys in env.get_state_desc()['body_pos'].keys():
    #     print(keys)
    # print(env.action_space.tolist())
total_reward = 0.0
