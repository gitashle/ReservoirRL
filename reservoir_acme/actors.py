from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from reservoir_acme.tf.networks import RNNNetwork

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree

tfd = tfp.distributions


class ReservoirActor(core.Actor):
    """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
            self,
            policy_network: RNNNetwork,
            adder: adders.Adder = None,
            variable_client: tf2_variable_utils.VariableClient = None,
    ):
        """Initializes the actor.

        Args:
          policy_network: the policy to run.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the learner copy
            of the policy to the actor copy (in case they are separate).
        """

        # Store these for later use.
        self._epsilon = 0.99999
        self._adder = adder
        self._variable_client = variable_client
        self._policy_network = policy_network

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_obs = tf2_utils.add_batch_dim(observation)

        # Forward the policy network.
        policy_output = self._policy_network(batched_obs)

        # If the policy network parameterises a distribution, sample from it.
        def maybe_sample(output):
            if np.random.random() > self._epsilon:
                return output
            return np.random.randint(0, self._policy_network.output_size)
#            return output

        # policy_output = tree.map_structure(maybe_sample, policy_output)
        # Convert to numpy and squeeze out the batch dimension.
        # action = tf2_utils.to_numpy_squeeze(policy_output)
        self._epsilon *= 0.99999999

        return np.array(maybe_sample(policy_output), dtype=np.int32)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(
            self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
    ):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self):
        if self._variable_client:
            self._variable_client.update()



