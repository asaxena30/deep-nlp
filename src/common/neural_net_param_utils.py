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


def print_gradients(named_parameters):
    """prints the average/max gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "print_gradients(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for name, param in named_parameters:
        if param.requires_grad and ("bias" not in name) and param.grad is not None:
            layers.append(name)
            average_grad = param.grad.abs().mean()
            max_grad = param.grad.abs().max()
            ave_grads.append(average_grad)
            max_grads.append(max_grad)
            print("param name: " + name)
            print("average grad: " + str(average_grad))
            print("max grad: " + str(max_grad))


