import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import PiecewiseSchedule, LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE=1000000
LEARNING_STARTS=50000
LEARNING_FREQ=4
FRAME_HISTORY_LEN=4
TARGER_UPDATE_FREQ=10000
GRAD_NORM_CLIPPING=10

def stopping_criterion(n):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return lambda env:  get_wrapper_by_name(env, "Monitor").get_total_steps() >= n

def main(env, num_timesteps):
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    # define learning rate and exploration schedules below
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2,  5e-5 * lr_multiplier),
    ], outside_value=5e-5 * lr_multiplier)

    optimizer = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(eps=1e-4),
        lr_schedule=lr_schedule
    )

    exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6, 0.1),
        (num_iterations / 2, 0.01),
    ], outside_value=0.01)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion(num_timesteps),
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        grad_norm_clipping=GRAD_NORM_CLIPPING
    )

if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, task.max_timesteps)
