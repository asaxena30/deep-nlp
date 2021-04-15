import torch
from torch.nn.init import normal_
from torch.nn.init import xavier_normal_, kaiming_normal_
from torch.utils.tensorboard import SummaryWriter

tensorboard_param_scalar_prefix: str = "param/"
tensorboard_activation_scalar_prefix: str = "activation/"


def xavier_normal_weight_init(named_parameters) -> None:
    for name, param in named_parameters:
        if 'weight' in name:
            xavier_normal_(param)


def kaiming_normal_weight_init(named_parameters) -> None:
    for name, param in named_parameters:
        if 'weight' in name:
            kaiming_normal_(param)


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
            mean_val = param.data.mean()
            max_val = param.data.max()
            print("param name: " + name)
            print("average grad abs: " + str(average_grad))
            print("max grad abs: " + str(max_grad))
            print("mean value: " + str(mean_val))
            print("max value" + str(max_val))


# note, specifying the type of named_parameters as Iterable[Tuple[str, Parameter]] etc. leads to the following error:
# TypeError: 'ABCMeta' object is not subscriptable
def add_parameter_stats_to_summary_writer(summary_writer: SummaryWriter, named_parameters, iteration: int):
    """prints the average/max gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "print_gradients(self.model.named_parameters())" to visualize the gradient flow"""
    for name, param in named_parameters:
        if param.requires_grad and ("bias" not in name) and param.grad is not None:
            resolved_name = tensorboard_param_scalar_prefix + name
            summary_writer.add_scalar(resolved_name + "_param_average_grad", param.grad.mean(), iteration)
            summary_writer.add_scalar(resolved_name + "_param_max_grad", param.grad.max(), iteration)
            summary_writer.add_scalar(resolved_name + "_param_min_grad", param.grad.min(), iteration)
            summary_writer.add_scalar(resolved_name + "_param_mean_value", param.data.mean(), iteration)
            summary_writer.add_scalar(resolved_name + "_param_median_value", param.data.median(), iteration)
            summary_writer.add_scalar(resolved_name + "_param_max_value", param.data.max(), iteration)
            summary_writer.add_scalar(resolved_name + "_param_variance", param.data.var(), iteration)


def add_tensor_stats_to_summary_writer(summary_writer: SummaryWriter, tensor: torch.Tensor, name: str,
                                       iteration: int, use_histogram=False):
    loggable_tensor = tensor.detach() if tensor.requires_grad else tensor
    resolved_name = tensorboard_activation_scalar_prefix + name

    if use_histogram:
        summary_writer.add_histogram(tag=resolved_name + "_histogram", values=loggable_tensor.data,
                                     global_step=iteration)
    else:
        summary_writer.add_scalar(resolved_name + "_mean_value", loggable_tensor.data.mean(), iteration)
        summary_writer.add_scalar(resolved_name + "_abs_mean_value", loggable_tensor.data.abs().mean(), iteration)
        summary_writer.add_scalar(resolved_name + "_max_value", loggable_tensor.data.max(), iteration)
        summary_writer.add_scalar(resolved_name + "_median_value", loggable_tensor.data.median(), iteration)
        summary_writer.add_scalar(resolved_name + "_min_value", loggable_tensor.data.min(), iteration)
        summary_writer.add_scalar(resolved_name + "_variance", loggable_tensor.data.var(), iteration)
