import gym
import torch.optim as optim

from dqn_model import DQN_RAM
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_ram_env, get_wrapper_by_name
from utils.schedule import PiecewiseSchedule, LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 1
TARGER_UPDATE_FREQ = 10000
GRAD_NORM_CLIPPING=10
EPS = 1e-4

def main(env, num_timesteps=int(4e7)):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2,  5e-5 * lr_multiplier),
    ], outside_value=5e-5 * lr_multiplier)

    optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(eps=EPS),
        lr_schedule=lr_schedule
    )

    exploration_schedule = PiecewiseSchedule([
        (0, 0.2),
        (1e6, 0.1),
        (num_iterations / 2, 0.01),
    ], outside_value=0.01)

    dqn_learing(
        env=env,
        q_func=DQN_RAM,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
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
    env = gym.make('Pong-ram-v0')

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_ram_env(env, seed)

    main(env)
