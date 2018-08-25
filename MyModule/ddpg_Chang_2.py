from __future__ import division
from collections import deque
import os
import warnings

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
# from rl.random import OrnsteinUhlenbeckProcess
from MyModule import random_process_Chang
from rl.util import *
import json
import pandas as pd

current_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_path,"../action_new.csv")
actionData_new=pd.read_csv(path,header=1)
label_new = ['bifemsh_l','gastroc_l','gastrocM_l','glut_max1_l','glut_max2_l',
    'glut_max3_l','glmed1','glmed2','glmed3','rect_fem_l','semimem_l','semiten_','soleus_r',
    'tibant_l','vaslat','vasmed_l']
af_new = actionData_new.fillna(0)
a_new = af_new.values.tolist()



def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent_Chang_2(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001,
                 param_noise=None, normalize_returns=False, normalize_observations=True,
                 observation_range=(-5., 5.),adaptive_param_noise=True,
                 adaptive_param_noise_policy_threshold=.1, **kwargs):
        # if hasattr(actor.output, '__len__') and len(actor.output) > 1:
        #     raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        # if hasattr(critic.output, '__len__') and len(critic.output) > 1:
        #     raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        # if critic_action_input not in critic.input:
        #     raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        # if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
        #     raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGAgent_Chang_2, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.param_noise = param_noise
        self.observation_range = observation_range

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def initialSample_action_new(self,experiment_act):
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
        # random_init = OrnsteinUhlenbeckProcess(theta=.15, mu=0.5, sigma=.2, size=env.get_action_space_size())

        action += self.random_process.sample()
        action = np.clip(action,0,1)
        return action

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        # self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor = self.actor
        self.target_actor.compile(optimizer='sgd', loss='mse')
        # self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic = self.critic
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer=actor_optimizer, loss=clipped_error, metrics=actor_metrics)

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(critic_inputs)

        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output))
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(critic_inputs + [K.learning_phase()],
                                             [self.actor(critic_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                critic_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(critic_inputs, [self.actor(critic_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def setup_param_noise(self, target_d=0.2, tol=1e-3, max_steps=1000):
        assert self.param_noise is not None
        experiences = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        state1_batch = []

        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)

        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)

        batch = state1_batch
        # Configure perturbed actor.
        orig_weights = self.actor.get_weights() #change for keras
        orig_act = self.actor.predict_on_batch(batch).flatten()
        # print(orig_act)

        sigma_min = 0.
        sigma_max = 50
        sigma = sigma_max
        step = 0
        while step < max_steps:
            weights = [w + np.random.normal(scale=sigma, size=np.shape(w)).astype('float32')
                       for w in orig_weights]
            self.actor.set_weights(weights)
            new_act = self.actor.predict_on_batch(batch).flatten()
            d = np.sqrt(np.mean(np.square(new_act - orig_act)))
            #distance between non-perturbed and perturbed policy

            dd = d - target_d
            if np.abs(dd) < tol:
                break

            # too big sigma
            if dd > 0:
                sigma_max = sigma
            # too small sigma
            else:
                sigma_min = sigma
            sigma = sigma_min + (sigma_max - sigma_min) / 2
            step += 1
        # print(step)


    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.random_process is not None:
            if self.rollout:
                noise = self.random_process.sample()
                assert noise.shape == action.shape
                action += noise
                # print('noise')
            elif self.action_noise:
                noise = self.random_process.sample()
                assert noise.shape == action.shape
                action += noise
                # print('noise')

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if self.training:
            # excecute when training is false
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics
        else: #training = false
            # print("update")
        # Train the network on a single stochastic batch.
            can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
            if can_train_either and self.step % self.train_interval == 0:
                experiences = self.memory.sample(self.batch_size)
                assert len(experiences) == self.batch_size

                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = []
                reward_batch = []
                action_batch = []
                terminal1_batch = []
                state1_batch = []
                for e in experiences:
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)
                # print(state0_batch)
                # Prepare and validate parameters.
                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                action_batch = np.array(action_batch)
                assert reward_batch.shape == (self.batch_size,)
                assert terminal1_batch.shape == reward_batch.shape
                assert action_batch.shape == (self.batch_size, self.nb_actions)

                # Update critic, if warm up is over.
                if self.step > self.nb_steps_warmup_critic:
                    target_actions = self.target_actor.predict_on_batch(state1_batch)
                    assert target_actions.shape == (self.batch_size, self.nb_actions)
                    if len(self.critic.inputs) >= 3:
                        state1_batch_with_action = state1_batch[:]
                    else:
                        state1_batch_with_action = [state1_batch]
                    state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                    target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                    assert target_q_values.shape == (self.batch_size,)

                    # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                    # but only for the affected output units (as given by action_batch).
                    discounted_reward_batch = self.gamma * target_q_values
                    discounted_reward_batch *= terminal1_batch
                    assert discounted_reward_batch.shape == reward_batch.shape
                    targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                    # Perform a single batch update on the critic network.
                    if len(self.critic.inputs) >= 3:
                        state0_batch_with_action = state0_batch[:]
                    else:
                        state0_batch_with_action = [state0_batch]
                    state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                    metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                    # print(self.critic.metrics_names)
                    if self.processor is not None:
                        metrics += self.processor.metrics

                # Update actor, if warm up is over.
                if self.step > self.nb_steps_warmup_actor:
                    # TODO: implement metrics for actor
                    if len(self.actor.inputs) >= 2:
                        inputs = state0_batch[:]
                    else:
                        inputs = [state0_batch]
                    if self.uses_learning_phase:
                        inputs += [self.training]
                    action_values = self.actor_train_fn(inputs)[0]
                    assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()
        # print(metrics)
        return metrics

    def train(self,env,nallsteps):
        nb_max_episode_steps = env.time_limit
        nb_max_start_steps = 20
        rollout_steps = 6
        training_steps = 1

        log_interval=10000
        max_steps = nallsteps
        visualize = False
        total_reward = 0
        done = False

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        episode_reward_log =[]

        action_repetition = 1
        param_noise_prob = 0.5

        # create buffer to store action and observations
        states_buffer = []
        action_buffer = []
        self.action_noise = False
        self.rollout = False
        print (self.training)
        self.training = True
        try:
            if observation == None:
                episode_step = np.int16(0)
                episode_reward = np.float32(0)
                episode_real_reward = np.float32(0)
                self.random_process.reset_states()
                # print(self.random_process.current_sigma)
                # observation = env.reset()
                # to start new simulations
                action = self.initialSample_action_new(a_new)
                # action = env.action_space.sample()
                action = np.clip(action,0,1)
                observation = env.reset()
                v = np.array(observation).reshape((env.observation_space.shape[0]))
                states_buffer.append([v])
                action_buffer.append(action)
                self.rollout = True
            while self.step < max_steps:
                    # start of a new episode
                    # callbacks.on_episode_begin(episode)

                    # initialize parameters
                    for _ in range(rollout_steps):
                        self.training = True
                        if self.step > self.nb_steps_warmup_actor:
                            self.action_noise = np.random.rand() < 1 - param_noise_prob
                            if not self.action_noise:
                                # print('perturb')
                                self.setup_param_noise(self.random_process.current_sigma)
                                # print(self.random_process.current_sigma)
                                self.rollout = False
                            else:
                                self.rollout = True

                        # add initialize parameters for the models
                        v = np.array(observation).reshape((env.observation_space.shape[0]))
                        action = self.forward(v)
                        # print(action[0])
                        observation, reward, done, info = env.step(action)
                        # self.memory.append(self.recent_observation, self.recent_action, reward, terminal=done,
                        #                    training=self.training)
                        self.backward(reward, terminal=done)
                        self.step += 1
                        episode_step += 1
                        episode_reward += reward
                        episode_real_reward += info['original_reward']
                        # print(self.step)
                        if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:                        # Force a terminal state.
                            done = True

                        if done:
                            action = self.forward(v)
                            self.backward(0., terminal=False)
                            warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the')
                            # observation = deepcopy(env.reset())
                            print(episode_reward, ' steps=',episode_step,' ',self.step,'/',max_steps)
                            print(episode_real_reward,'real')
                            episode_reward_log.append(episode_real_reward)
                            episode += 1
                            episode_step = np.int16(0)
                            episode_reward = np.float32(0)
                            episode_real_reward = np.float32(0)
                            self.random_process.reset_states()
                            # reset
                            action = self.initialSample_action_new(a_new)
                            # action = env.action_space.sample()
                            action = np.clip(action,0,1)
                            observation = env.reset()


                            # break
                    for _ in range(training_steps):
                        # print(observation)
                        self.training = False
                        self.rollout = False
                        self.action_noise = False
                        # assert episode_reward is not None
                        # assert episode_step is not None
                        assert observation is not None
                        metrics = self.backward(reward, terminal=done)
                        # print(metrics)
                        # print('initialization')
                        # This is were all of the work happens. We first perceive and compute the action
                                # (forward step) and then use the reward to improve (backward step).
                        # v = np.array(observation).reshape((env.observation_space.shape[0]))
                        # action = self.forward(v)
                        # observation, reward, done, info = env.step(action.tolist())
                        # episode_step += 1
                        # self.step += 1
                        # episode_reward += reward
                        # episode_real_reward += info['original_reward']
                        # states_buffer.append([v])
                        # action_buffer.append(action)


                            # experience log in agent.backward
                        # print(episode_reward)
                        # print(episode_real_reward)
                        # step_logs = {
                        #     'action': action,
                        #     'observation': observation,
                        #     'reward': reward,
                        #     'metrics': metrics,
                        #     'episode': episode,
                        #     'info': info,
                        # }

                        # print(env.reward(),'/',max_steps)

                        # if done:
                        #     # We are in a terminal state but the agent hasn't yet seen it. We therefore
                        #     # perform one more forward-backward call and simply ignore the action before
                        #     # resetting the environment. We need to pass in `terminal=False` here since
                        #     # the *next* state, that is the state of the newly reset environment, is
                        #     # always non-terminal by convention.
                        #     states_buffer.append([v])
                        #     action_buffer.append(action)
                        #
                        #     # states_np = np.asarray(states_buffer,dtype=np.float32)
                        #     episode_logs = {
                        #     'episode_reward': episode_reward,
                        #     'nb_episode_steps': episode_step,
                        #     'nb_steps': self.step,
                        #     }
                        #     print(episode_reward, ' steps=',episode_step,' ',self.step,'/',max_steps)
                        #     print(episode_real_reward,'real')
                        #     episode_reward_log.append(episode_real_reward)
                        #     episode += 1
                        #     observation = None
                        #     episode_step = np.int16(0)
                        #     episode_reward = np.float32(0)
                        #     episode_real_reward = np.float32(0)
                        #
                        #
                        #     # del states_np
                        #     del states_buffer[:]
                        #     del action_buffer[:]

        except KeyboardInterrupt:
            did_abort = True
            self.save_weights(args.model, overwrite=True)

        log_filename = '/Users/liuchang/test_log.json'
        with open(log_filename, "w") as write_file:
            json.dump(episode_reward_log, write_file)
        # return episode_reward_log
        # self.save_weights(args.model, overwrite=True)
