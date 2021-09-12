"""Run DQN on Atari."""

import sys
sys.path.append('../')

import dm_env
import gym
import acme
import functools
from absl import app
from absl import flags
from acme import wrappers
from acme.utils import loggers

sys.path.append('../../')

from reservoir_acme.tf import networks
from reservoir_acme import EnvironmentTimeLoop
from reservoir_acme.agents.tf import reservoir

flags.DEFINE_string('level', 'BattleZone-v0', 'Which Atari level to play.')
flags.DEFINE_integer('num_episodes', 20000, 'Number of episodes to train for.')
flags.DEFINE_string('results_dir', 'atari_time/environment_loop/', 'CSV results directory.')

FLAGS = flags.FLAGS


def make_environment(evaluation: bool = False) -> dm_env.Environment:
    env = gym.make(FLAGS.level, full_action_space=False)
    max_episode_len = 108_000 if evaluation else 50_000

    return wrappers.wrap_all(env, [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            to_float=True,
            max_episode_len=max_episode_len,
            zero_discount_on_life_loss=True,
        ),
        wrappers.SinglePrecisionWrapper,
    ])


def main(_):
    env = make_environment()
    env_spec = acme.make_environment_spec(env)
    network = networks.RNNNetwork(input_size=512, output_size=env_spec.actions.num_values, recurrent_units=2000,
                                  percentage_of_recurrent_neurons_to_train=30)
    agent_terminal_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
    env_loop_terminal_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

    agent_csv_logger = loggers.CSVLogger(directory=FLAGS.level, label='agent_csv', time_delta=10.)
    env_loop_csv_logger = loggers.CSVLogger(directory=FLAGS.level, label='env_loop_csv', time_delta=10.)

    agent_logger = loggers.Dispatcher([agent_terminal_logger, agent_csv_logger])
    env_loop_logger = loggers.Dispatcher([env_loop_terminal_logger, env_loop_csv_logger])

    agent = reservoir.ReservoirDQN(environment_spec=env_spec,
                                   network=network,
                                   logger=agent_logger,
                                   batch_size=200,
                                   checkpoint_subpath='reservoir_checkpoints/')
    loop = EnvironmentTimeLoop(env, agent, logger=env_loop_logger)
    loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
