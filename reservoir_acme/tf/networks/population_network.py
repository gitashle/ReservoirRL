import torch
import time
import pickle
import numpy as np
import torch
from copy import deepcopy
from torch import nn

USE_CUDA = False 
device = torch.device(
    "cuda" if USE_CUDA and torch.cuda.is_available() and 'GeForce' not in torch.cuda.get_device_name(0) else "cpu")

TRAIN_RECURRENT_NEURONS = False
TRAIN_OUTPUT_NEURONS = True
RESTING_STATE_TIMESTEPS = 100
MIN_RNN_CLIP = -0.4
MAX_RNN_CLIP = 0.5
MIN_OUTPUT_CLIP = -0.4
MAX_OUTPUT_CLIP = 0.8


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()
        self.name = 'activation'

    def forward(self, input_x):
        return input_x.view(input_x.size(0), -1)

class VisionModule(nn.Module):

    def __init__(self):
        super(VisionModule, self).__init__()
        self._network = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            Flatten(),
            nn.Linear(1600, 1000),
            nn.Linear(1000, 512)
        )

    def forward(self, x):
        return self._network(x)

        
class RNNNetwork(object):

    def __init__(self, input_size, recurrent_units, output_size, connection_probability=0.8,
                 learning_rate=1e-4, percentage_of_recurrent_neurons_to_train=100):

        self.input_size = input_size
        self.recurrent_units = recurrent_units
        self.output_size = output_size
        self.connection_prob = connection_probability
        self.tau = 25  # ms
        self.gain_of_network = 5
        self.state_of_RNN = np.zeros((self.recurrent_units, self.recurrent_units))
        self.input_weights = []
        self.recurrent_weights = []
        self.output_weights = []
        self.time_step = 1
        self.alpha = learning_rate
        self.train_dt = 20
        self.bias_current = 1e-4
        self.error_plt = 0
        self.threshold = 0.4
        self.set_weights()
        self.percentage_of_recurrent_neurons_to_train = percentage_of_recurrent_neurons_to_train
        self.train_rnn_neurons = torch.randperm(self.recurrent_units)[
                                 :int(self.recurrent_units * (self.percentage_of_recurrent_neurons_to_train / 100))]
        self.vision_module = VisionModule()

        self.timesteps = 20
#        self.parameters = list(self.vision_module.parameters()) + [self.input_weights, self.recurrent_weights, self.output_weights]
        self.parameters = list(self.vision_module.parameters()) + [self.output_weights]

    def set_weights(self):
        # self.input_weights = torch.empty((self.input_size, self.recurrent_units), device=device, dtype=torch.float,
        #                                  requires_grad=False)
        self.input_weights = torch.ones((self.input_size, self.recurrent_units), device=device, dtype=torch.float,
                                        requires_grad=False) * 0.1
        torch.nn.init.normal_(self.input_weights, 0, 2)
        input_weight_mask = np.zeros(shape=(self.input_size, self.recurrent_units))
        size_col_to_mask = int(self.recurrent_units / self.input_size)

        for i in range(self.input_size):
            if i == 0:
                input_weight_mask[i, i * size_col_to_mask: size_col_to_mask + (i * size_col_to_mask)] = 1
            else:
                input_weight_mask[i, 1 + (i * size_col_to_mask): size_col_to_mask + (i * size_col_to_mask)] = 1

        self.input_weights = self.input_weights * torch.tensor(input_weight_mask, device=device, dtype=torch.float)
        self.input_weights = self.input_weights.T

        # recurrent
        self.recurrent_weights = torch.empty((self.recurrent_units, self.recurrent_units), device=device,
                                             dtype=torch.float, requires_grad=False)
        std_rec = 1 / np.sqrt(self.gain_of_network * self.recurrent_units)
        torch.nn.init.normal_(self.recurrent_weights, 0, std_rec)
        recurrent_weight_mask = np.ones(shape=(self.recurrent_units, self.recurrent_units))
        recurrent_weight_mask[recurrent_weight_mask < (1 - self.connection_prob)] = 0
        self.recurrent_weights = self.recurrent_weights * torch.tensor(recurrent_weight_mask, device=device,
                                                                       dtype=torch.float) * self.gain_of_network
        self.recurrent_weights[1: (self.recurrent_units + 1): self.recurrent_units * self.recurrent_units] = 0
        # output
        self.output_weights = torch.empty((self.output_size, self.recurrent_units), device=device, dtype=torch.float,
                                          requires_grad=True)
        self.output_weights = torch.nn.init.normal_(self.output_weights, 0, 1 / np.sqrt(self.recurrent_units))

    def _convert_continous_state(self, features):
        batch, feature_size = features.shape
        _features = features.repeat(1, self.timesteps).reshape(batch, self.timesteps, -1)
        return torch.cat([_features, torch.zeros((batch, 50, _features.shape[-1]))], 1)

    def reset(self):
        self.input_cont_signal = None
        self.output_cont_signal = None
        self.reservoir_state = None

    def _update_input_cont_signal(self, input_signal):
        if self.input_cont_signal is None:
            self.input_cont_signal = input_signal
        else:
            self.input_cont_signal = torch.cat([input_signal, self.input_cont_signal], 1)

    def _update_output_cont_signal(self, output_signal):
        raise NotImplementedError

    def get_reservoir_state(self, batch):
        if self.reservoir_state is None:
            self.reservoir_state = torch.zeros(size=(self.recurrent_units, batch), device=device) + self.bias_current
        return self.reservoir_state

    def __call__(self, *args, **kwargs):
        training = kwargs.get('training', False)
        _input = torch.FloatTensor(args[0].numpy().transpose(0, 3, 1, 2))
        # add the _input to the class instance
        features = self.vision_module(_input)
        continous_feature = self._convert_continous_state(features).to(device)  # axis-0 : time , axis-1 : feature
        self._update_input_cont_signal(continous_feature)
        result = self.forward(continous_feature)
        output = torch.mean(result, 1)
        return torch.argmax(output).to('cpu').detach().type(torch.int32).numpy() if not training else output

    def forward(self, input_signal):
        batch, timesteps, features = input_signal.shape
        self.get_reservoir_state(batch)
        output_signal = torch.zeros(size=(batch, timesteps, self.output_size), device=device)
        recurrent_neurons_firing_rate = torch.zeros((self.recurrent_units, batch), device=device)
        for t in range(timesteps):
            R = torch.matmul(self.recurrent_weights, recurrent_neurons_firing_rate)
            I = torch.matmul(input_signal[:, t, :], self.input_weights.T).T
            self.reservoir_state = ((self.reservoir_state * (1 - (1 / self.tau))) + ((1 / self.tau) * (R + I)))
            recurrent_neurons_firing_rate = torch.tanh(self.reservoir_state)
            output_signal[:, t, :] = torch.matmul(self.output_weights, recurrent_neurons_firing_rate).view(batch, -1)
        return output_signal

    def load_weights(self, filepath):
        with open('{}/rnn_weights.pik'.format(filepath), 'rb') as io:
            k = pickle.load(io)
        self.input_weights = k['input_weights']
        self.recurrent_weights = k['rnn_weights']
        self.output_weights = k['output_weights']
        self.recurrent_weights = self.recurrent_weights.to(device)
        self.input_weights = self.input_weights.to(device)
        self.output_weights = self.output_weights.to(device)

    def save_weights(self, filepath):
        try:
            with open('{}/rnn_weights.pik'.format(filepath), 'wb') as io:
                pickle.dump({'input_weights': self.input_weights.detach().cpu(),
                             'rnn_weights': self.recurrent_weights.detach().cpu(),
                             'output_weights': self.output_weights.detach().cpu()}, io)
        except:
            print('failed to save weights')
            pass


if __name__ == '__main__':
    network = RNNNetwork(input_size=100, recurrent_units=200, output_size=1)
    # input_signal = torch.rand(size=(1, 84, 84, 4))
    # output = network(input_signal)
    input_signal = torch.rand(size=(20, 84, 84, 4))
    output = network(input_signal)
