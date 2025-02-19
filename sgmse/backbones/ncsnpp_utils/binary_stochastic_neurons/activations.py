import torch
import torch.nn as nn

from binary_stochastic_neurons.utils import Hardsigmoid, RoundST, BernoulliST
from binary_stochastic_neurons.distributions import Bernoulli as BernoulliREINFORCE
from binary_stochastic_neurons.distributions import Round as RoundREINFORCE


class DeterministicBinaryActivation(nn.Module):

    def __init__(self, estimator="ST"):
        super(DeterministicBinaryActivation, self).__init__()

        assert estimator in ["ST", "REINFORCE"]

        self.estimator = estimator
        self.act = Hardsigmoid()

        if self.estimator == "ST":
            self.binarizer = RoundST
        elif self.estimator == "REINFORCE":
            self.binarizer = RoundREINFORCE

    def forward(self, input):
        x, slope = input
        x = self.act(slope * x)
        x = self.binarizer(x)
        if self.estimator == "REINFORCE":
            x = x.sample()
        return x


class StochasticBinaryActivation(nn.Module):

    def __init__(self, estimator="ST"):
        super(StochasticBinaryActivation, self).__init__()

        assert estimator in ["ST", "REINFORCE"]

        self.estimator = estimator
        self.act = Hardsigmoid()

        if self.estimator == "ST":
            self.binarizer = BernoulliST
        elif self.estimator == "REINFORCE":
            self.binarizer = BernoulliREINFORCE

    def forward(self, input):
        x, slope = input
        probs = self.act(slope * x)
        out = self.binarizer(probs)
        if self.estimator == "REINFORCE":
            out = out.sample()
        return out
