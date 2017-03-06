"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.schedule import LinearSchedule
from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

# Check if the parameters of the model update accordingly.
def check_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2.0)
        total_norm += param_norm ** 2.0
    total_norm = total_norm ** (1.0 / 2.0)
    return total_norm


def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)

    # Construct optimizer with adaptive learning rate
    # https://discuss.pytorch.org/t/adaptive-learning-rate/320
    # Currently in torch, we have create a new optimizer every time when we want to adjust learning rate dynamically.
    def construct_optimizer(t):
        lr = optimizer_spec.lr_schedule.value(t)
        return optimizer_spec.constructor(Q.parameters(), lr=lr, **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    for t in count():
        ### Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        last_idx = replay_buffer.store_frame(last_obs)
        # encode_recent_observation will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # recent_observations: shape(img_h, img_w, frame_history_len) are input to to the model
        recent_observations = replay_buffer.encode_recent_observation().transpose(2, 0, 1)
        # recent_observations = replay_buffer.encode_recent_observation()

        # Choose random action if not yet start learning
        if t > learning_starts:
            action = select_epilson_greedy_action(target_Q, recent_observations, t)[0, 0]
        else:
            action = random.randrange(num_actions)
        # Advance one step
        obs, reward, done, _ = env.step(action)
        replay_buffer.store_effect(last_idx, action, reward, done)
        # Resets the environment when reaching an episode boundary.
        if done:
            obs = env.reset()
        last_obs = obs

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
            obs_batch = Variable(torch.from_numpy(obs_batch.transpose(0, 3, 1, 2)).type(dtype) / 255.0)
            # obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch.transpose(0, 3, 1, 2)).type(dtype) / 255.0, volatile=True)
            # next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0, volatile=True)
            done_mask = torch.from_numpy(done_mask)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
                done_mask = done_mask.cuda()

            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
            # Compute next Q value, based on which acion gives max Q values
            next_max_Q_values = Variable(torch.zeros(batch_size).type(dtype))
            # # Detach variable from the current graph since we don't want gradients to propagated
            next_max_Q_values[done_mask == 0] = target_Q(next_obs_batch).detach().max(1)[0]
            # Compute Bellman error, use huber loss to mitigate outlier impact
            bellman_error = F.smooth_l1_loss(current_Q_values, rew_batch + (gamma * next_max_Q_values))
            # Run backward pass and clip the gradient
            Q.zero_grad()
            bellman_error.backward()

            if check_norm(Q) > grad_norm_clipping:
                print('Before clipping gradient:')
                print('total_norm: ', check_norm(Q))
                nn.utils.clip_grad_norm(Q.parameters(), grad_norm_clipping)
                print('After clipping gradient:')
                print('total_norm: ', check_norm(Q))
            # Perfom the update
            optimizer = construct_optimizer(t)
            optimizer.step()
            # print('After update Q:')
            # check_norm(Q)
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                # print('Before update target:')
                # check_norm(target_Q)
                for target_param, param in zip(target_Q.parameters(), Q.parameters()):
                    target_param.data = param.data.clone()
                # print('After update target:')
                # check_norm(target_Q)

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
