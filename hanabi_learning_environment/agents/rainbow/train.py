# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
# This code has been adapted by F Bredell to accomodate artificial conventions.
#
"""The entry point for running a Rainbow conventions agent on Hanabi."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from third_party.dopamine import logger
import datetime

import run_experiment
from small_hanabi_conventions_encoder import simple_combined_encoder, simple_transfer_encoder, standalone_encoder
from hanabi_conventions_encoder import simple_combined_encoder_full, simple_combined_encoder_full_v2, simple_official_rules_based_encoder_2p, simple_official_rules_based_encoder

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").') #remember to set path to config file

flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

flags.DEFINE_string('base_dir', '/tmp/full_hanabi_rainbow_convention_encouded_official_3p_non_lenient_'+current_time,
                    'Base directory to host all required sub-directories.') #tmp is used for now, but is strongly recommended to change to other directory, otherwise agents will be lost on pc reset. 

# flags.DEFINE_string('base_dir', None,
#                     'Base directory to host all required sub-directories.') #default

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', '',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')


def launch_experiment():
  """Launches the experiment.

  Specifically:
  - Load the gin configs and bindings.
  - Initialize the Logger object.
  - Initialize the environment.
  - Initialize the observation stacker.
  - Initialize the convention layers/encoders.
  - Initialize the agent.
  - Reload from the latest checkpoint, if available, and initialize the
    Checkpointer object.
  - Run the experiment.
  """
  if FLAGS.base_dir == None:
    raise ValueError('--base_dir is None: please provide a path for '
                     'logs and checkpoints.')

  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

  environment = run_experiment.create_environment()
  obs_stacker = run_experiment.create_obs_stacker(environment)

  #----------------------------------Small Hanabi conventions---------------------------  
  # Simplify later with a check for env version to auto select which encoder to use
  # convention_encoder = simple_combined_encouder(environment)  #basic
  # convention_encoder = simple_combined_encouder_v2(environment)  #basic
  # convention_encoder = simple_transfer_encouder(environment)  #transfer
  #--------------------------------------------------------------------------

  #--------------------------------Full Hanabi conventions-------------------------------
  # convention_encoder = simple_official_rules_based_encoder_2p(environment)
  convention_encoder = simple_official_rules_based_encoder(environment)
  #-------------------------------------------------------------------------

  agent = run_experiment.create_agent(environment, obs_stacker, convention_encoder)

  checkpoint_dir = '{}/checkpoints'.format(FLAGS.base_dir)
  start_iteration, experiment_checkpointer = (
      run_experiment.initialize_checkpointing(agent,
                                              experiment_logger,
                                              checkpoint_dir,
                                              FLAGS.checkpoint_file_prefix))

  run_experiment.run_experiment(agent, environment, start_iteration,
                                obs_stacker,
                                experiment_logger, experiment_checkpointer,
                                checkpoint_dir, convention_encoder = convention_encoder,
                                logging_file_prefix=FLAGS.logging_file_prefix)


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment()

if __name__ == '__main__':
  app.run(main)
