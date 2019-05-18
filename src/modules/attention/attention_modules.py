import torch

from torch.nn import Module
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_


class SelfAttention(Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.attention_weights_lhs = torch.nn.Parameter(kaiming_normal_(torch.Tensor(1, 1, input_size)))
        self.attention_weights_rhs = torch.nn.Parameter(kaiming_normal_(torch.Tensor(1, 1, input_size)))

    def forward(self, input_instance_batch_as_tensor: torch.Tensor):
        """
        :param input_instance_batch_as_tensor: (N, SEQUENCE_LENGTH, INPUT_SIZE)
        :return: self-attention enriched outputs (N, SEQUENCE_LENGTH, INPUT_SIZE)
        """
        assert input_instance_batch_as_tensor.shape[2] == self.input_size

        left_hand_side_tensors = input_instance_batch_as_tensor * self.attention_weights_lhs
        right_hand_side_tensors = input_instance_batch_as_tensor * self.attention_weights_rhs

        # final_attention_weights has dim (N, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        final_attention_weights = F.softmax(torch.matmul(left_hand_side_tensors, torch.transpose(right_hand_side_tensors, 1, 2)), dim = 2)

        assert final_attention_weights.shape[1] == final_attention_weights.shape[2]

        return torch.matmul(final_attention_weights, input_instance_batch_as_tensor)


class BidirectionalAttention(Module):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.attention_weights_lhs = torch.nn.Parameter(kaiming_normal_(torch.Tensor(1, 1, input_size)))
        self.attention_weights_rhs = torch.nn.Parameter(kaiming_normal_(torch.Tensor(1, 1, input_size)))

    def forward(self, lhs_input_instance_batch_as_tensor: torch.Tensor, rhs_input_instance_batch_as_tensor: torch.Tensor):
        """
        :param lhs_input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :param rhs_input_instance_batch_as_tensor: (N, RHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, LHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) and
        (N, RHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) respectively
        """

        assert lhs_input_instance_batch_as_tensor.shape[2] == self.input_size
        assert rhs_input_instance_batch_as_tensor.shape[2] == self.input_size

        left_hand_side_tensors = lhs_input_instance_batch_as_tensor * self.attention_weights_lhs
        right_hand_side_tensors = rhs_input_instance_batch_as_tensor * self.attention_weights_rhs

        # final_attention_weights has dim (N, lhs_SEQUENCE_LENGTH, rhs_SEQUENCE_LENGTH)
        final_attention_weights_unnormalized = torch.matmul(left_hand_side_tensors, torch.transpose(right_hand_side_tensors, 1, 2))

        # softmax-for-lhs => each lhs vector will be replaced with a weighted combination of rhs
        final_attention_weights_softmaxed_for_lhs = F.softmax(final_attention_weights_unnormalized, dim = 2)

        # softmax-for-rhs => each rhs vector will be replaced with a weighted combination of lhs
        final_attention_weights_softmaxed_for_rhs = F.softmax(final_attention_weights_unnormalized, dim = 1)

        return torch.cat([lhs_input_instance_batch_as_tensor,
                          torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor)], dim = 2), \
               torch.cat([rhs_input_instance_batch_as_tensor,
                          torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)], dim = 2)

#
# class LuongAttention(Module):
#     """
#            inputs:
#                input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
#                the word vector
#        """
#
#     def __init__(self, lhs_input_size: int, rhs_input_size: int):
#         super().__init__()
#         self.lhs_weights = torch.nn.Linear(lhs_input_size, )




