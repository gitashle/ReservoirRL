"""Run DQN on Atari."""

import functools
from absl import app
from absl import flags
import acme
from acme import wrappers
from reservoir_acme.tf import networks
from acme.utils import loggers
import dm_env
import gym

from acme.agents.tf import dqn
# flags.DEFINE_string('level', 'BeamRider-v0', 'Which Atari level to play.')
#flags.DEFINE_string('level', 'Breakout-v0', 'Which Atari level to play.')
flags.DEFINE_string('level', 'Robotank-v0', 'Which Atari level to play.')
flags.DEFINE_integer('num_episodes', 30000, 'Number of episodes to train for.')
# flags.DEFINE_string('results_dir', 'atari_time/environment_loop/', 'CSV results directory.')

FLAGS = flags.FLAGS


def make_environment(evaluation: bool = False) -> dm_env.Environment:
    env = gym.make(FLAGS.level, full_action_space=True)
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

    network = networks.DQNAtariNetwork(env_spec.actions.num_values)
    agent_terminal_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
    env_loop_terminal_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

    agent_csv_logger = loggers.CSVLogger(directory='atari_time/logs/Robotank_dqn', label='agent_csv', time_delta=10.)
    env_loop_csv_logger = loggers.CSVLogger(directory='atari_time/logs/Robotank_dqn', label='env_loop_csv', time_delta=10.)

    agent_logger = loggers.Dispatcher([agent_terminal_logger, agent_csv_logger])
    env_loop_logger = loggers.Dispatcher([env_loop_terminal_logger, env_loop_csv_logger])


    agent = dqn.DQN(environment_spec=env_spec,
                    network=network,
                    logger=agent_logger,
                    max_replay_size=10000)
    loop = acme.EnvironmentLoop(env, agent, logger=env_loop_logger)
    loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
