import torch

from torch.nn.init import xavier_normal_
from torch.nn.init import normal_


def xavier_normal_weight_init(named_parameters) -> None:
    for name, param in named_parameters:
        if 'weight' in name:
            xavier_normal_(param)


def normal_weight_init(named_parameters, mean: int = 0, standard_deviation: int = 0.1) -> None:
    for name, param in named_parameters:
        if 'weight' in name:
            normal_(param, mean, standard_deviation)


def init_lstm_forget_gate_biases(lstm: torch.nn.LSTM, value):
    for names in lstm._all_weights:
        for name in filter(lambda name: "bias" in name, names):
            bias = getattr(lstm, name)
            bias_vector_size = bias.size(0)
            start, end = bias_vector_size // 4, bias_vector_size // 2
            bias.data[start:end].fill_(value)


def print_gradients(named_parameters):
    """prints the average/max gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "print_gradients(self.model.named_parameters())" to visualize the gradient flow"""
    for name, param in named_parameters:
        if param.requires_grad and ("bias" not in name) and param.grad is not None:
            average_grad = param.grad.abs().mean()
            max_grad = param.grad.abs().max()
            mean_val = param.data.abs().mean()
            max_val = param.data.abs().max()
            print("param name: " + name)
            print("average grad: " + str(average_grad))
            print("max grad: " + str(max_grad))
            print("mean value" + str(mean_val))
            print("max value" + str(max_val))


