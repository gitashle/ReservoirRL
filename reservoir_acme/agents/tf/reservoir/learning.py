# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN learner implementation."""

import time
from typing import Dict, List

import acme
from acme.adders import reverb as adders
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import torch
from torch import optim
import trfl


class ReservoirDQNLearner(acme.Learner, tf2_savers.TFSaveable):
    """DQN learner.

    This is the learning component of a DQN agent. It takes a dataset as input
    and implements update functionality to learn from this dataset. Optionally
    it takes a replay client as well to allow for updating of priorities.
    """

    def __init__(
            self,
            network: snt.Module,
            target_network: snt.Module,
            discount: float,
            importance_sampling_exponent: float,
            learning_rate: float,
            target_update_period: int,
            dataset: tf.data.Dataset,
            huber_loss_parameter: float = 1.,
            replay_client: reverb.TFClient = None,
            counter: counting.Counter = None,
            logger: loggers.Logger = None,
            checkpoint: bool = True,
    ):
        """Initializes the learner.

        Args:
          network: the online Q network (the one being optimized)
          target_network: the target Q critic (which lags behind the online net).
          discount: discount to use for TD updates.
          importance_sampling_exponent: power to which importance weights are raised
            before normalizing.
          learning_rate: learning rate for the q-network update.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          dataset: dataset to learn from, whether fixed or from a replay buffer (see
            `acme.datasets.reverb.make_dataset` documentation).
          huber_loss_parameter: Quadratic-linear boundary for Huber loss.
          replay_client: client to replay to allow for updating priorities.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        # Internalise agent components (replay buffer, networks, optimizer).
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self.dataset = dataset
        self._network = network
        self._target_network = target_network
        self._optimizer = snt.optimizers.Adam(learning_rate)
        self._replay_client = replay_client

        # Internalise the hyperparameters.
        self._discount = discount
        self._target_update_period = target_update_period
        self._importance_sampling_exponent = importance_sampling_exponent
        self._huber_loss_parameter = huber_loss_parameter

        # Learner state.
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # Internalise logging/counting objects.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        # Create a snapshotter object.
        self._snapshotter = None

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None
        self.time_to_act = []

        self.optim = optim.Adam(self._network.parameters, lr=0.01)

    
    def _step(self) -> Dict[str, tf.Tensor]:
        """Do a step of SGD and update the priorities."""

        # Pull out the data needed for updates/priorities.
        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
        keys, probs = inputs.info[:2]
        q_tm1 = self._network(o_tm1, training=True)
        q_t_value = self._target_network(o_t, training=True).detach()
        q_t_selector = self._network(o_t, training=True).detach()

        r_t = torch.FloatTensor(r_t.numpy())
        r_t += 0.000001
        r_t = torch.clamp(r_t, -1., 1.)
        d_t = d_t * self._discount
        r_t = torch.FloatTensor(r_t)
        r_t = r_t.to(q_tm1.device)
        d_t = torch.FloatTensor(d_t.numpy())
        d_t = d_t.to(q_tm1.device)
        a_tm1 = torch.FloatTensor(a_tm1.numpy())
        a_tm1 = a_tm1.to(q_tm1.device)
#            probs = torch.FloatTensor(probs.numpy())
#            probs = probs.to(q_tm1.device)

        best_action = torch.argmax(q_t_selector, 1)
        double_q_bootstrapped = torch.gather(q_t_value, 1, best_action.view(1,-1))
        target = r_t + d_t * double_q_bootstrapped
        qa_tm1 = torch.gather(q_tm1, 1, a_tm1.type(torch.int64).view(1, -1))
        td_error = target - qa_tm1
        loss = 0.5 * torch.square(td_error)
        # Get the importance weights.
        importance_weights = 1./ probs
        importance_weights **= self._importance_sampling_exponent
        importance_weights /= tf.reduce_max(importance_weights)
        importance_weights = torch.FloatTensor(importance_weights.numpy())
        importance_weights = importance_weights.to(q_tm1.device)
        loss *= importance_weights
        loss = torch.mean(loss, 1)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Update the priorities in the replay buffer.
        if self._replay_client:
            priorities = tf.cast(tf.abs(td_error.view(-1).detach().cpu().numpy()), tf.float64)
            self._replay_client.update_priorities(
                table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

        # Periodically update the target network.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(self._network.parameters,
                                 self._target_network.parameters):
                dest = src
        self._num_steps.assign_add(1)

        # Report loss & statistics for logging.
        fetches = {
            'loss': loss.detach().cpu().numpy(),
        }

        return fetches

    def step(self):
        # Do a batch of SGD.
        result = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        result.update(counts)

        # Snapshot and attempt to write logs.
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(result)

    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        return tf2_utils.to_numpy(self._variables)

    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'network': self._network,
            'target_network': self._target_network,
            'optimizer': self._optimizer,
            'num_steps': self._num_steps
        }
