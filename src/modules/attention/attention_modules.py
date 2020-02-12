import math

import torch

from torch.nn import Module
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, normal_, xavier_normal_


class SymmetricBidirectionalAttention(Module):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
    """
    def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, scale_dot_products = True,
                 activation = torch.nn.ReLU, linear_layer_weight_init = kaiming_normal_, use_guided_attention = False):
        super().__init__()
        self.input_size = input_size
        self.linear_layer = torch.nn.Linear(input_size, input_size)
        linear_layer_weight_init(self.linear_layer.weight)

        self.return_with_inputs_concatenated = return_with_inputs_concatenated
        self.dot_product_scale_factor = math.sqrt(input_size) if scale_dot_products else 1
        self.activation = activation()
        self.use_guided_attention = use_guided_attention

        # if self.use_guided_attention:
        #     self.guidance_tensor_weights = torch.nn.Parameter(torch.ones(input_size))

    def forward(self, lhs_input_instance_batch_as_tensor: torch.Tensor,
                rhs_input_instance_batch_as_tensor: torch.Tensor, guidance_tensor: torch.Tensor = None):
        """
        :param lhs_input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :param rhs_input_instance_batch_as_tensor: (N, RHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :param guidance_tensor: (N, lhs_SEQUENCE_LENGTH, rhs_SEQUENCE_LENGTH)
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, LHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) and
        (N, RHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) respectively
        """

        # assert lhs_input_instance_batch_as_tensor.shape[2] == self.input_size
        # assert rhs_input_instance_batch_as_tensor.shape[2] == self.input_size

        left_hand_side_tensors = self.activation(self.linear_layer(lhs_input_instance_batch_as_tensor))
        right_hand_side_tensors = self.activation(self.linear_layer(rhs_input_instance_batch_as_tensor))

        # multiply the weighted guidance_tensor to either one of lhs/rhs if self.use_guided_attention is true
        if self.use_guided_attention:
            # assert guidance_tensor.shape[0] == lhs_input_instance_batch_as_tensor.shape[0]
            # left_hand_side_tensors = left_hand_side_tensors * F.softmax(guidance_tensor * self.guidance_tensor_weights, dim = 2)
            left_hand_side_tensors = left_hand_side_tensors * guidance_tensor

        # final_attention_weights has dim (N, lhs_SEQUENCE_LENGTH, rhs_SEQUENCE_LENGTH)
        final_attention_weights_unnormalized = \
            torch.matmul(left_hand_side_tensors, torch.transpose(right_hand_side_tensors, 1, 2))/self.dot_product_scale_factor


        # softmax-for-lhs => each lhs vector will be replaced with a weighted combination of rhs
        final_attention_weights_softmaxed_for_lhs = F.softmax(final_attention_weights_unnormalized, dim = 2)

        # softmax-for-rhs => each rhs vector will be replaced with a weighted combination of lhs
        final_attention_weights_softmaxed_for_rhs = F.softmax(final_attention_weights_unnormalized, dim = 1)

        if self.return_with_inputs_concatenated:
            return torch.cat([lhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor)], dim = 2), \
                   torch.cat([rhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)], dim = 2)

        return torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor), \
               torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)


class BidirectionalAttention(Module):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
    """
    def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, activation = torch.nn.Tanh,
                 scale_dot_products: bool = True, linear_layer_weight_init = xavier_normal_):
        super().__init__()
        self.input_size = input_size
        self.linear_layer_lhs = torch.nn.Linear(input_size, input_size, bias = False)
        self.linear_layer_rhs = torch.nn.Linear(input_size, input_size, bias = False)

        linear_layer_weight_init(self.linear_layer_lhs.weight)
        linear_layer_weight_init(self.linear_layer_rhs.weight)

        self.activation = activation() if activation is not None else lambda x: x

        self.return_with_inputs_concatenated = return_with_inputs_concatenated
        self.dot_product_scale_factor = math.sqrt(input_size) if scale_dot_products else 1
        # self.dot_product_scale_factor = 1

    def forward(self, lhs_input_instance_batch_as_tensor: torch.Tensor, rhs_input_instance_batch_as_tensor: torch.Tensor):
        """
        :param lhs_input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :param rhs_input_instance_batch_as_tensor: (N, RHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, LHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) and
        (N, RHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) respectively
        """

        assert lhs_input_instance_batch_as_tensor.shape[2] == self.input_size
        assert rhs_input_instance_batch_as_tensor.shape[2] == self.input_size

        left_hand_side_tensors = self.activation(self.linear_layer_lhs(lhs_input_instance_batch_as_tensor))
        right_hand_side_tensors = self.activation(self.linear_layer_rhs(rhs_input_instance_batch_as_tensor))

        # final_attention_weights has dim (N, lhs_SEQUENCE_LENGTH, rhs_SEQUENCE_LENGTH)
        # final_attention_weights_unnormalized = torch.matmul(left_hand_side_tensors, torch.transpose(right_hand_side_tensors, 1, 2))/self.dot_product_scale_factor

        final_attention_weights_unnormalized = torch.matmul(left_hand_side_tensors, torch.transpose(right_hand_side_tensors, 1, 2))

        # softmax-for-lhs => each lhs vector will be replaced with a weighted combination of rhs
        final_attention_weights_softmaxed_for_lhs = F.softmax(final_attention_weights_unnormalized, dim = 2)

        # softmax-for-rhs => each rhs vector will be replaced with a weighted combination of lhs
        # final_attention_weights_softmaxed_for_rhs = F.softmax(final_attention_weights_unnormalized, dim = 1)
        final_attention_weights_softmaxed_for_rhs = F.softmax(final_attention_weights_unnormalized/self.dot_product_scale_factor, dim = 1)

        if self.return_with_inputs_concatenated:
            return torch.cat([lhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor)], dim = 2), \
                   torch.cat([rhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)], dim = 2)

        return torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor), \
               torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)


class BidirectionalAttentionSimplified(Module):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
    """
    def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, activation = torch.nn.Tanh,
                 scale_dot_products: bool = True):
        super().__init__()
        self.input_size = input_size
        self.lhs_weights = torch.nn.Parameter(torch.ones(input_size))
        self.rhs_weights = torch.nn.Parameter(torch.ones(input_size))

        self.activation = activation() if activation is not None else lambda x: x

        self.return_with_inputs_concatenated = return_with_inputs_concatenated
        self.dot_product_scale_factor = math.sqrt(input_size) if scale_dot_products else 1
        # self.dot_product_scale_factor = 1

    def forward(self, lhs_input_instance_batch_as_tensor: torch.Tensor, rhs_input_instance_batch_as_tensor: torch.Tensor):
        """
        :param lhs_input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :param rhs_input_instance_batch_as_tensor: (N, RHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, LHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) and
        (N, RHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) respectively
        """

        assert lhs_input_instance_batch_as_tensor.shape[2] == self.input_size
        assert rhs_input_instance_batch_as_tensor.shape[2] == self.input_size

        left_hand_side_tensors = self.activation(self.lhs_weights * lhs_input_instance_batch_as_tensor)
        right_hand_side_tensors = self.activation(self.rhs_weights * rhs_input_instance_batch_as_tensor)

        # final_attention_weights has dim (N, lhs_SEQUENCE_LENGTH, rhs_SEQUENCE_LENGTH)
        final_attention_weights_unnormalized = torch.matmul(left_hand_side_tensors, torch.transpose(right_hand_side_tensors, 1, 2))/self.dot_product_scale_factor

        # softmax-for-lhs => each lhs vector will be replaced with a weighted combination of rhs
        final_attention_weights_softmaxed_for_lhs = F.softmax(final_attention_weights_unnormalized, dim = 2)

        # softmax-for-rhs => each rhs vector will be replaced with a weighted combination of lhs
        final_attention_weights_softmaxed_for_rhs = F.softmax(final_attention_weights_unnormalized, dim = 1)

        if self.return_with_inputs_concatenated:
            return torch.cat([lhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor)], dim = 2), \
                   torch.cat([rhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)], dim = 2)

        return torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor), \
               torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)


class SymmetricSelfAttention(SymmetricBidirectionalAttention):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
            return_with_inputs_concatenated: if true, the response returned will be a tuple with inputs concatenated and hence the
            3rd dimension will be 2 * input_size
            scale_dot_products: if True, the attention dot products will be divided by sq-root(input_size), default True
    """
    def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, scale_dot_products = True,
                 activation = torch.nn.ReLU, linear_layer_weight_init = kaiming_normal_):
        super().__init__(input_size, return_with_inputs_concatenated, scale_dot_products, activation = activation,
                         linear_layer_weight_init = linear_layer_weight_init)

    def forward(self, input_instance_batch_as_tensor: torch.Tensor):
        """
        :param input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, SEQUENCE_LENGTH, 2 * INPUT_SIZE)
        """
        return super().forward(input_instance_batch_as_tensor, input_instance_batch_as_tensor)[0]


class AsymmetricSelfAttention(BidirectionalAttention):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
            return_with_inputs_concatenated: if true, the response returned will be a tuple with inputs concatenated and hence the
            3rd dimension will be 2 * input_size
            scale_dot_products: if True, the attention dot products will be divided by sq-root(input_size), default True
    """
    def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, scale_dot_products = True,
                 activation = torch.nn.ReLU, linear_layer_weight_init = kaiming_normal_):
        super().__init__(input_size = input_size, return_with_inputs_concatenated = return_with_inputs_concatenated,
                         scale_dot_products = scale_dot_products, activation = activation,
                         linear_layer_weight_init = linear_layer_weight_init)

    def forward(self, input_instance_batch_as_tensor: torch.Tensor):
        """
        :param input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, SEQUENCE_LENGTH, 2 * INPUT_SIZE)
        """
        result = super().forward(input_instance_batch_as_tensor, input_instance_batch_as_tensor)
        return (result[0] + result[1]) / 2


class BidafAttention(Module):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
    """
    def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, activation = torch.nn.Tanh,
                 linear_layer_weight_init = xavier_normal_):
        super().__init__()
        self.input_size = input_size
        self.linear_layer = torch.nn.Linear(input_size * 3, 1)

        linear_layer_weight_init(self.linear_layer.weight)

        self.activation = activation() if activation is not None else lambda x: x

        self.return_with_inputs_concatenated = return_with_inputs_concatenated

    def forward(self, lhs_input_instance_batch_as_tensor: torch.Tensor, rhs_input_instance_batch_as_tensor: torch.Tensor):
        """
        :param lhs_input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :param rhs_input_instance_batch_as_tensor: (N, RHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, LHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) and
        (N, RHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) respectively
        """

        assert lhs_input_instance_batch_as_tensor.shape[2] == self.input_size
        assert rhs_input_instance_batch_as_tensor.shape[2] == self.input_size

        lhs_input_instance_batch_as_tensor_unsqueezed = lhs_input_instance_batch_as_tensor.unsqueeze(dim = 2)
        rhs_input_instance_batch_as_tensor_unsqueezed = rhs_input_instance_batch_as_tensor.unsqueeze(dim = 1)

        lhs_rhs_elemwise_product = lhs_input_instance_batch_as_tensor_unsqueezed * rhs_input_instance_batch_as_tensor_unsqueezed

        # final_attention_weights has dim (N, lhs_SEQUENCE_LENGTH, rhs_SEQUENCE_LENGTH)
        final_attention_weights_unnormalized = self.activation(self.linear_layer(torch.cat([lhs_rhs_elemwise_product,
                                                                                            lhs_input_instance_batch_as_tensor_unsqueezed.expand_as(lhs_rhs_elemwise_product),
                                                                                            rhs_input_instance_batch_as_tensor_unsqueezed.expand_as(lhs_rhs_elemwise_product)],
                                                                                           dim = 3))).squeeze(dim = 3)


        # softmax-for-lhs => each lhs vector will be replaced with a weighted combination of rhs
        final_attention_weights_softmaxed_for_lhs = F.softmax(final_attention_weights_unnormalized, dim = 2)

        # softmax-for-rhs => each rhs vector will be replaced with a weighted combination of lhs
        final_attention_weights_softmaxed_for_rhs = F.softmax(final_attention_weights_unnormalized, dim = 1)

        if self.return_with_inputs_concatenated:
            return torch.cat([lhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor)], dim = 2), \
                   torch.cat([rhs_input_instance_batch_as_tensor,
                              torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)], dim = 2)

        return torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor), \
               torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)


# class ModifiedBidafAttention(Module):
#     """
#         inputs:
#             input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
#             the word vector
#     """
#     def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, activation = torch.nn.Tanh,
#                  linear_layer_weight_init = xavier_normal_):
#         super().__init__()
#         self.input_size = input_size
#         self.linear_layer = torch.nn.Linear(input_size, 1)
#         self.linear_layer_lhs = torch.nn.Linear(input_size, 1)
#         self.linear_layer_rhs = torch.nn.Linear(input_size, 1)
#
#         linear_layer_weight_init(self.linear_layer.weight)
#         linear_layer_weight_init(self.linear_layer_lhs.weight)
#         linear_layer_weight_init(self.linear_layer_rhs.weight)
#
#         self.activation = activation() if activation is not None else lambda x: x
#
#         self.return_with_inputs_concatenated = return_with_inputs_concatenated
#
#     def forward(self, lhs_input_instance_batch_as_tensor: torch.Tensor, rhs_input_instance_batch_as_tensor: torch.Tensor):
#         """
#         :param lhs_input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
#         :param rhs_input_instance_batch_as_tensor: (N, RHS_SEQUENCE_LENGTH, INPUT_SIZE)
#         :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, LHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) and
#         (N, RHS_SEQUENCE_LENGTH, 2 * INPUT_SIZE) respectively
#         """
#
#         assert lhs_input_instance_batch_as_tensor.shape[2] == self.input_size
#         assert rhs_input_instance_batch_as_tensor.shape[2] == self.input_size
#
#         lhs_input_instance_batch_as_tensor_unsqueezed = lhs_input_instance_batch_as_tensor.unsqueeze(dim = 2)
#         rhs_input_instance_batch_as_tensor_unsqueezed = rhs_input_instance_batch_as_tensor.unsqueeze(dim = 1)
#
#         lhs_rhs_elemwise_product = lhs_input_instance_batch_as_tensor_unsqueezed * rhs_input_instance_batch_as_tensor_unsqueezed
#
#         # final_attention_weights has dim (N, lhs_SEQUENCE_LENGTH, rhs_SEQUENCE_LENGTH)
#         final_attention_weights_unnormalized = self.activation(self.linear_layer(torch.cat([lhs_rhs_elemwise_product,
#                                                                                             lhs_input_instance_batch_as_tensor_unsqueezed.expand_as(lhs_rhs_elemwise_product),
#                                                                                             rhs_input_instance_batch_as_tensor_unsqueezed.expand_as(lhs_rhs_elemwise_product)],
#                                                                                            dim = 3))).squeeze(dim = 3)
#
#
#         # softmax-for-lhs => each lhs vector will be replaced with a weighted combination of rhs
#         final_attention_weights_softmaxed_for_lhs = F.softmax(final_attention_weights_unnormalized, dim = 2)
#
#         # softmax-for-rhs => each rhs vector will be replaced with a weighted combination of lhs
#         final_attention_weights_softmaxed_for_rhs = F.softmax(final_attention_weights_unnormalized, dim = 1)
#
#         if self.return_with_inputs_concatenated:
#             return torch.cat([lhs_input_instance_batch_as_tensor,
#                               torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor)], dim = 2), \
#                    torch.cat([rhs_input_instance_batch_as_tensor,
#                               torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)], dim = 2)
#
#         return torch.matmul(final_attention_weights_softmaxed_for_lhs, rhs_input_instance_batch_as_tensor), \
#                torch.matmul(final_attention_weights_softmaxed_for_rhs.transpose(1, 2), lhs_input_instance_batch_as_tensor)


class GuidedSelfAttention(SymmetricBidirectionalAttention):
    """
        inputs:
            input_size: size of the left-hand-side input tensor. for eg. for NLP, this would be the size of
            the word vector
            return_with_inputs_concatenated: if true, the response returned will be a tuple with inputs concatenated and hence the
            3rd dimension will be 2 * input_size
            scale_dot_products: if True, the attention dot products will be divided by sq-root(input_size), default True
    """
    def __init__(self, input_size: int, return_with_inputs_concatenated: bool = True, scale_dot_products = True,
                 activation = torch.nn.ReLU, linear_layer_weight_init = kaiming_normal_):
        super().__init__(input_size = input_size, return_with_inputs_concatenated = return_with_inputs_concatenated,
                         scale_dot_products = scale_dot_products, activation = activation,
                         linear_layer_weight_init = linear_layer_weight_init, use_guided_attention = True)

    def forward(self, input_instance_batch_as_tensor: torch.Tensor, guidance_tensor: torch.Tensor):
        """
        :param input_instance_batch_as_tensor: (N, LHS_SEQUENCE_LENGTH, INPUT_SIZE)
        :param guidance_tensor: (N), required
        :return: Tuple[Tensor, Tensor]. attention enriched tensors of size (N, SEQUENCE_LENGTH, 2 * INPUT_SIZE) or (N, SEQUENCE_LENGTH, INPUT_SIZE)
        depending upon the value of return_with_inputs_concatenated
        """
        return super().forward(lhs_input_instance_batch_as_tensor = input_instance_batch_as_tensor,
                        rhs_input_instance_batch_as_tensor = input_instance_batch_as_tensor,
                        guidance_tensor = guidance_tensor)[0]

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




