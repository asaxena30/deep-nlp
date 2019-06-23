from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertLayerNorm
from torch.nn.modules.module import Module
import torch.nn
from typing import Dict, List

from src.data.instance.instance import QAInstanceWithAnswerSpan
from src.modules.attention.attention_modules import BidirectionalAttention, SelfAttention
from src.util.vector_encoding_utils import build_index_tensor_for_tokenized_sentences

BERT_BASE_HIDDEN_SIZE: int = 768
BERT_LARGE_HIDDEN_SIZE: int = 1024


def init_bert_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean = 0.0, std = 0.02)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class BertQuestionAnsweringModule(Module):

    def __init__(self, bert_model_name = 'bert-base-uncased', batch_first = True, device = None,
                 named_param_weight_initializer = None):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        init_bert_weights(self.bert_model)
        self.bert_model_name = bert_model_name
        self.lstm = torch.nn.LSTM(self.__resolve_bert_model_size(), 300, bidirectional = True, dropout = 0,
                                  batch_first = batch_first)
        self.linear_layer_for_answer_start_index = torch.nn.Linear(600, 1)
        self.linear_layer_for_answer_end_index = torch.nn.Linear(600, 1)
        self.softmax = torch.nn.Softmax(dim = (1 if batch_first else 0))
        self.device = device

        # we don't want to initialize the bert_model weights
        if named_param_weight_initializer is not None:
            named_param_weight_initializer(self.lstm.named_parameters())
            named_param_weight_initializer(self.linear_layer_for_answer_start_index.named_parameters())
            named_param_weight_initializer(self.linear_layer_for_answer_end_index.named_parameters())

    def forward(self, input_ids, token_type_ids, attention_mask, dummy_checkpointing_variable = None,
                output_all_encoded_layers = False, skip_tuning_bert_model=False):

        if skip_tuning_bert_model:
            with torch.no_grad():
                bert_model_output: torch.Tensor = self.bert_model(input_ids = input_ids, token_type_ids = token_type_ids,
                                                                  attention_mask = attention_mask,
                                                                  output_all_encoded_layers = output_all_encoded_layers)
        else:
            bert_model_output: torch.Tensor = self.bert_model(input_ids = input_ids, token_type_ids = token_type_ids,
                                                              attention_mask = attention_mask,
                                                              output_all_encoded_layers = output_all_encoded_layers)
        lstm_output = self.lstm(bert_model_output[0], self.__init_lstm_hidden_and_cell_state(
                                                              input_ids.size()[0]))
        # start_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_start_index(lstm_output[0]))
        # end_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_end_index(lstm_output[0]))

        start_index_scores_tensor = self.linear_layer_for_answer_start_index(lstm_output[0])
        end_index_scores_tensor = self.linear_layer_for_answer_end_index(lstm_output[0])

        return start_index_scores_tensor, end_index_scores_tensor

    def __resolve_bert_model_size(self):
        return BERT_BASE_HIDDEN_SIZE if self.bert_model_name.startswith('bert-base') else BERT_LARGE_HIDDEN_SIZE

    def __init_lstm_hidden_and_cell_state(self, batch_size: int):
        return (
            torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.lstm.hidden_size,
                        device = self.device),
            torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.lstm.hidden_size,
                        device = self.device))


class BertQuestionAnsweringModuleSimplified(Module):
    """counterpart of BertQuestionAnsweringModule that avoids using an LSTM"""

    def __init__(self, bert_model_name = 'bert-base-uncased', batch_first = True, device = None,
                 named_param_weight_initializer = None):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        init_bert_weights(self.bert_model)
        self.bert_model_name = bert_model_name
        self.bert_model_size = self.__resolve_bert_model_size()
        self.linear_layer_for_answer_start_index = torch.nn.Linear(self.bert_model_size, 1)
        self.linear_layer_for_answer_end_index = torch.nn.Linear(self.bert_model_size, 1)
        self.softmax = torch.nn.Softmax(dim = (1 if batch_first else 0))
        self.device = device

        # we don't want to initialize the bert_model weights
        if named_param_weight_initializer is not None:
            named_param_weight_initializer(self.linear_layer_for_answer_start_index.named_parameters())
            named_param_weight_initializer(self.linear_layer_for_answer_end_index.named_parameters())

    def forward(self, input_ids, token_type_ids, attention_mask, dummy_checkpointing_variable = None,
                output_all_encoded_layers = False, skip_tuning_bert_model=False):

        if skip_tuning_bert_model:
            with torch.no_grad():
                bert_model_output: torch.Tensor = self.bert_model(input_ids = input_ids,
                                                                  token_type_ids = token_type_ids,
                                                                  attention_mask = attention_mask,
                                                                  output_all_encoded_layers = output_all_encoded_layers)
        else:
            bert_model_output: torch.Tensor = self.bert_model(input_ids = input_ids, token_type_ids = token_type_ids,
                                                              attention_mask = attention_mask,
                                                              output_all_encoded_layers = output_all_encoded_layers)
        #
        # start_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_start_index(bert_model_output[0]))
        # end_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_end_index(bert_model_output[0]))

        start_index_scores_tensor = self.linear_layer_for_answer_start_index(bert_model_output[0])
        end_index_scores_tensor = self.linear_layer_for_answer_end_index(bert_model_output[0])

        return start_index_scores_tensor, end_index_scores_tensor

    def __resolve_bert_model_size(self):
        return BERT_BASE_HIDDEN_SIZE if self.bert_model_name.startswith('bert-base') else BERT_LARGE_HIDDEN_SIZE


class QAModuleWithAttentionNoBert(Module):
    """
        a relatively wide model which employs bi-lstms, bidirectional_attention and self-attention parallely
        and merges the outputs with a linear layer on top to determine the start and end indices of an answer span

        :param embedding
        :param token_to_index_dict
        :param embedding_index_for_unknown_words

        Got:
        start index accuracy: 0.145748987854251
        end index accuracy: 0.15975033738191632
        exact match accuracy: 0.0772604588394062 (increases to 0.08+ if the cross entropy loss is scaled by a factor which denotes the degree of overlap between
        the actual and the chosen answer span)
        total answers: 5928
    """
    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__()
        self.embedding = embedding
        self.lstm_for_question_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                        bidirectional = True, batch_first = True, bias = False)
        self.lstm_for_passage_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)
        self.bidirection_attention_module = BidirectionalAttention(self.embedding.embedding_dim)

        self.self_attention_module_1 = SelfAttention(self.embedding.embedding_dim)
        self.self_attention_module_2 = SelfAttention(self.embedding.embedding_dim)

        self.final_question_encoder = torch.nn.LSTM(self.embedding.embedding_dim * 6, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 8, 1, bias = False)
        self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 8, 1, bias = False)

        self.device = device

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):

        question_batch = self.embedding(question_batch_index_tensor)
        passage_batch = self.embedding(passage_batch_index_tensor)

        question_and_passage_batch_concatenated = torch.cat([question_batch, passage_batch], dim = 1)

        # the pytorch lstm outputs: output, (h_n, c_n)
        question_lstm_output = self.lstm_for_question_encoding(question_batch, self.__init_lstm_hidden_and_cell_state(
            question_batch.size()[0], self.embedding.embedding_dim))[0]

        passage_lstm_output = self.lstm_for_passage_encoding(passage_batch, self.__init_lstm_hidden_and_cell_state(
            passage_batch.size()[0], self.embedding.embedding_dim))[0]

        question_bidirectional_attention_output, passage_bidirectional_attention_output = \
            self.bidirection_attention_module(question_batch, passage_batch)

        # the result is now (N, SEQUENCE_LENGTH, 4 * input_size) where SEQUENCE_LENGTH is passage/question length
        question_lstm_and_bi_att_output_concatenated = torch.cat([question_lstm_output, question_bidirectional_attention_output], dim = 2)
        passage_lstm_and_bi_att_output_concatenated = torch.cat([passage_lstm_output, passage_bidirectional_attention_output], dim = 2)

        # # this is the same dimension as the (N, input question + passage concatenated, INPUT_SIZE)
        self_attention_output_level_1 = self.self_attention_module_1(question_and_passage_batch_concatenated)
        self_attention_output_level_2 = self.self_attention_module_2(self_attention_output_level_1)

        # splitting the self-attention enriched question and passage and extracting the latter. the passage output is
        # (N, passage length, INPUT_SIZE)
        question_splitted_after_self_attention, passage_splitted_after_self_attention = torch.split(self_attention_output_level_2, split_size_or_sections = [question_batch.shape[1],
                                                                                                passage_batch.shape[1]], dim = 1)

        # (N, SEQUENCE_LENGTH, 6 * input_size)
        final_concatenated_passage_output = torch.cat([passage_lstm_and_bi_att_output_concatenated, passage_splitted_after_self_attention,
                                                       passage_batch], dim = 2)
        final_concatenated_question_output = torch.cat([question_lstm_and_bi_att_output_concatenated, question_splitted_after_self_attention,
                                                        question_batch], dim = 2)

        # simply taking the final lstm state which should be (N, SEQUENCE_LENGTH , 2 * INPUT_SIZE)
        final_question_encoding_raw = self.final_question_encoder(final_concatenated_question_output,
                                                                  self.__init_lstm_hidden_and_cell_state(
            final_concatenated_question_output.size()[0], self.embedding.embedding_dim))[1][0]
        final_question_encoding = torch.cat(torch.chunk(torch.transpose(final_question_encoding_raw, 0, 1), chunks = 2, dim = 1), dim = 2)
        final_question_encoding_scaled_to_passage_length = torch.cat([final_question_encoding] * final_concatenated_passage_output.shape[1], dim = 1)

        passage_outputs_enriched_with_final_question_encoding = torch.cat([final_concatenated_passage_output,
                                                                              final_question_encoding_scaled_to_passage_length], dim = 2)

        final_outputs_for_start_index = self.linear_layer_for_start_index(passage_outputs_enriched_with_final_question_encoding)
        final_outputs_for_end_index = self.linear_layer_for_end_index(passage_outputs_enriched_with_final_question_encoding)

        return final_outputs_for_start_index, final_outputs_for_end_index

    def __init_lstm_hidden_and_cell_state(self, batch_size: int, hidden_size: int, is_bidirectional: bool = True,
                                          num_layers: int = 1):
        return (
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device),
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device))


class QAModuleWithAttentionNoBert2(Module):
    """
        Replaces self attention with bi-att modules on self, similar but not identical as the latter (refer to attention_modules.py)
        is a little less parameterized
    """
    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__()
        self.embedding = embedding
        self.lstm_for_question_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                        bidirectional = True, batch_first = True, bias = False)
        self.lstm_for_passage_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_final_question_embedding = torch.nn.Linear(self.embedding.embedding_dim * 2, self.embedding.embedding_dim)
        self.linear_layer_for_final_passage_embedding = torch.nn.Linear(self.embedding.embedding_dim * 2, self.embedding.embedding_dim)

        self.bidirectional_attention_module = BidirectionalAttention(self.embedding.embedding_dim)
        self.passage_to_passage_attention = BidirectionalAttention(self.embedding.embedding_dim)
        self.question_to_question_attention = BidirectionalAttention(self.embedding.embedding_dim)

        self.bidirectional_attention_module_2 = BidirectionalAttention(self.embedding.embedding_dim * 2,
                                                                       return_with_inputs_concatenated = False)

        self.final_question_encoder = torch.nn.LSTM(self.embedding.embedding_dim * 6, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 8, 1, bias = False)
        self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 8, 1, bias = False)

        self.device = device

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):

        question_batch = self.embedding(question_batch_index_tensor)
        passage_batch = self.embedding(passage_batch_index_tensor)

        # the pytorch lstm outputs: output, (h_n, c_n)
        question_lstm_output = self.lstm_for_question_encoding(question_batch, self.__init_lstm_hidden_and_cell_state(
            question_batch.size()[0], self.embedding.embedding_dim))[0]

        passage_lstm_output = self.lstm_for_passage_encoding(passage_batch, self.__init_lstm_hidden_and_cell_state(
            passage_batch.size()[0], self.embedding.embedding_dim))[0]

        question_bidirectional_attention_output, passage_bidirectional_attention_output = \
            self.bidirectional_attention_module(question_batch, passage_batch)

        # the result is now (N, SEQUENCE_LENGTH, 4 * input_size) where SEQUENCE_LENGTH is passage/question length
        question_lstm_and_bi_att_output_concatenated = torch.cat([question_lstm_output, question_bidirectional_attention_output], dim = 2)
        passage_lstm_and_bi_att_output_concatenated = torch.cat([passage_lstm_output, passage_bidirectional_attention_output], dim = 2)

        # each part of the tuples is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
        # input with the bi-att output
        passage_self_bi_att_output_lhs, passage_self_bi_att_output_rhs = self.passage_to_passage_attention(passage_batch, passage_batch)
        question_self_bi_att_output_lhs, question_self_bi_att_output_rhs = self.question_to_question_attention(question_batch, question_batch)

        # (N, length, 2 * INPUT_SIZE)
        passage_self_bi_att_output = (passage_self_bi_att_output_lhs + passage_self_bi_att_output_rhs)/2.0
        question_self_bi_att_output = (question_self_bi_att_output_lhs + question_self_bi_att_output_rhs)/2.0

        # (N, length, 2 * INPUT_SIZE) each
        question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
            self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)

        # (N, SEQUENCE_LENGTH, 6 * input_size)
        final_concatenated_passage_output = torch.cat([passage_lstm_and_bi_att_output_concatenated,
                                                       passage_bidirectional_attention_output_2], dim = 2)
        final_concatenated_question_output = torch.cat([question_lstm_and_bi_att_output_concatenated,
                                                        question_bidirectional_attention_output_2], dim = 2)

        # simply taking the final lstm state which should be (N, SEQUENCE_LENGTH , 2 * INPUT_SIZE)
        final_question_encoding_raw = self.final_question_encoder(final_concatenated_question_output,
                                                                  self.__init_lstm_hidden_and_cell_state(
            final_concatenated_question_output.size()[0], self.embedding.embedding_dim))[1][0]
        final_question_encoding = torch.cat(torch.chunk(torch.transpose(final_question_encoding_raw, 0, 1), chunks = 2, dim = 1), dim = 2)
        final_question_encoding_scaled_to_passage_length = torch.cat([final_question_encoding] * final_concatenated_passage_output.shape[1], dim = 1)

        passage_outputs_enriched_with_final_question_encoding = torch.cat([final_concatenated_passage_output,
                                                                              final_question_encoding_scaled_to_passage_length], dim = 2)

        final_outputs_for_start_index = self.linear_layer_for_start_index(passage_outputs_enriched_with_final_question_encoding)
        final_outputs_for_end_index = self.linear_layer_for_end_index(passage_outputs_enriched_with_final_question_encoding)

        return final_outputs_for_start_index, final_outputs_for_end_index

    def __init_lstm_hidden_and_cell_state(self, batch_size: int, hidden_size: int, is_bidirectional: bool = True,
                                          num_layers: int = 1):
        return (
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device),
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device))


class QAModuleWithAttentionNoBert3(Module):
    """
        Replaces self attention with bi-att modules on self, similar but not identical as the latter (refer to attention_modules.py)
        is a little less parameterized

        best so far:

        start index accuracy: 0.1946693657219973
        end index accuracy: 0.20917678812415655
        exact match accuracy: 0.12331309041835357
        total answers: 5928
    """
    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__()
        self.embedding = embedding
        self.lstm_for_question_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                        bidirectional = True, batch_first = True, bias = False)
        self.lstm_for_passage_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_final_question_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.embedding.embedding_dim)
        self.linear_layer_for_final_passage_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.embedding.embedding_dim)

        self.bidirectional_attention_module = BidirectionalAttention(self.embedding.embedding_dim)
        self.passage_to_passage_attention = BidirectionalAttention(self.embedding.embedding_dim)
        self.question_to_question_attention = BidirectionalAttention(self.embedding.embedding_dim)

        self.bidirectional_attention_module_2 = BidirectionalAttention(self.embedding.embedding_dim * 2,
                                                                       return_with_inputs_concatenated = False)

        self.final_question_encoder = torch.nn.LSTM(self.embedding.embedding_dim * 6, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 8, 1, bias = False)
        self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 8, 1, bias = False)

        self.device = device

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):

        question_batch = self.embedding(question_batch_index_tensor)
        passage_batch = self.embedding(passage_batch_index_tensor)

        # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
        question_lstm_output = self.lstm_for_question_encoding(question_batch, self.__init_lstm_hidden_and_cell_state(
            question_batch.size()[0], self.embedding.embedding_dim))[0]

        passage_lstm_output = self.lstm_for_passage_encoding(passage_batch, self.__init_lstm_hidden_and_cell_state(
            passage_batch.size()[0], self.embedding.embedding_dim))[0]

        # (N, seq_length, embedding_size)
        passage_embedding = self.linear_layer_for_final_passage_embedding(torch.cat([passage_batch, passage_lstm_output], dim = 2))
        question_embedding = self.linear_layer_for_final_question_embedding(torch.cat([question_batch, question_lstm_output], dim = 2))

        question_bidirectional_attention_output, passage_bidirectional_attention_output = \
            self.bidirectional_attention_module(question_embedding, passage_embedding)

        # the result is now (N, SEQUENCE_LENGTH, 4 * input_size) where SEQUENCE_LENGTH is passage/question length
        question_lstm_and_bi_att_output_concatenated = torch.cat([question_batch, question_embedding, question_bidirectional_attention_output], dim = 2)
        passage_lstm_and_bi_att_output_concatenated = torch.cat([passage_batch, passage_embedding, passage_bidirectional_attention_output], dim = 2)

        # each part of the tuples is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
        # input with the bi-att output
        passage_self_bi_att_output_lhs, passage_self_bi_att_output_rhs = self.passage_to_passage_attention(passage_embedding, passage_embedding)
        question_self_bi_att_output_lhs, question_self_bi_att_output_rhs = self.question_to_question_attention(question_embedding, question_embedding)

        # (N, length, 2 * INPUT_SIZE)
        passage_self_bi_att_output = (passage_self_bi_att_output_lhs + passage_self_bi_att_output_rhs)/2.0
        question_self_bi_att_output = (question_self_bi_att_output_lhs + question_self_bi_att_output_rhs)/2.0

        # (N, length, 2 * INPUT_SIZE) each
        question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
            self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)

        # (N, SEQUENCE_LENGTH, 6 * input_size)
        final_concatenated_passage_output = torch.cat([passage_lstm_and_bi_att_output_concatenated,
                                                       passage_bidirectional_attention_output_2], dim = 2)
        final_concatenated_question_output = torch.cat([question_lstm_and_bi_att_output_concatenated,
                                                        question_bidirectional_attention_output_2], dim = 2)

        # simply taking the final lstm state which should be (N, SEQUENCE_LENGTH , 2 * INPUT_SIZE)
        final_encoded_question_representation = self.final_question_encoder(final_concatenated_question_output,
                                              self.__init_lstm_hidden_and_cell_state(
                                                  final_concatenated_question_output.size()[0],
                                                  self.embedding.embedding_dim))
        final_question_encoding_raw = final_encoded_question_representation[1][0]
        final_question_encoding = torch.cat(torch.chunk(torch.transpose(final_question_encoding_raw, 0, 1), chunks = 2, dim = 1), dim = 2)
        final_question_encoding_scaled_to_passage_length = torch.cat([final_question_encoding] * final_concatenated_passage_output.shape[1], dim = 1)

        passage_outputs_enriched_with_final_question_encoding = torch.cat([final_concatenated_passage_output,
                                                                              final_question_encoding_scaled_to_passage_length], dim = 2)

        final_outputs_for_start_index = self.linear_layer_for_start_index(passage_outputs_enriched_with_final_question_encoding)
        final_outputs_for_end_index = self.linear_layer_for_end_index(passage_outputs_enriched_with_final_question_encoding)

        return final_outputs_for_start_index, final_outputs_for_end_index

    def __init_lstm_hidden_and_cell_state(self, batch_size: int, hidden_size: int, is_bidirectional: bool = True,
                                          num_layers: int = 1):
        return (
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device),
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device))


class QAModuleWithAttentionNoBert4(Module):
    """
        Replaces self attention with bi-att modules on self, similar but not identical as the latter (refer to attention_modules.py)
        is a little less parameterized. also precdes the final linear layer by another_lstm. This ie begin done to possibly make the
        final question encoder a bit more useful

        start index accuracy: 0.22351551956815116
        end index accuracy: 0.24358974358974358
        exact match accuracy: 0.1524966261808367
        total answers: 5928

        this model performing better than it's predecessor (#3 as of 2019-06-22) indicates that the final single linear layer
        alone can only do so much with an input which is embedding-size * 8. however preceding it with another LSTM
        which takes both question and passage into account and then squeezes the output to embedding-size * 2 helps.
        It's not clear how useful squeezing the output size is, but it does look like the addition of another LSTM
        captures dependencies better.

    """
    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__()
        self.embedding = embedding
        self.lstm_for_question_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                        bidirectional = True, batch_first = True, bias = False)
        self.lstm_for_passage_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_final_question_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.embedding.embedding_dim)
        self.linear_layer_for_final_passage_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.embedding.embedding_dim)

        self.bidirectional_attention_module = BidirectionalAttention(self.embedding.embedding_dim)
        self.passage_to_passage_attention = BidirectionalAttention(self.embedding.embedding_dim)
        self.question_to_question_attention = BidirectionalAttention(self.embedding.embedding_dim)

        self.bidirectional_attention_module_2 = BidirectionalAttention(self.embedding.embedding_dim * 2,
                                                                       return_with_inputs_concatenated = False)

        self.final_question_encoder = torch.nn.LSTM(self.embedding.embedding_dim * 6, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.lstm_for_final_question_and_passage_combination = torch.nn.LSTM(self.embedding.embedding_dim * 8, self.embedding.embedding_dim * 2,
                                                       bidirectional = False, batch_first = True, bias = False)

        self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 2, 1, bias = False)
        self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 2, 1, bias = False)

        self.device = device

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):

        question_batch = self.embedding(question_batch_index_tensor)
        passage_batch = self.embedding(passage_batch_index_tensor)

        # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
        question_lstm_output = self.lstm_for_question_encoding(question_batch, self.__init_lstm_hidden_and_cell_state(
            question_batch.size()[0], self.embedding.embedding_dim))[0]

        passage_lstm_output = self.lstm_for_passage_encoding(passage_batch, self.__init_lstm_hidden_and_cell_state(
            passage_batch.size()[0], self.embedding.embedding_dim))[0]

        # (N, seq_length, embedding_size)
        passage_embedding = self.linear_layer_for_final_passage_embedding(torch.cat([passage_batch, passage_lstm_output], dim = 2))
        question_embedding = self.linear_layer_for_final_question_embedding(torch.cat([question_batch, question_lstm_output], dim = 2))

        question_bidirectional_attention_output, passage_bidirectional_attention_output = \
            self.bidirectional_attention_module(question_embedding, passage_embedding)

        # the result is now (N, SEQUENCE_LENGTH, 4 * input_size) where SEQUENCE_LENGTH is passage/question length
        question_lstm_and_bi_att_output_concatenated = torch.cat([question_batch, question_embedding, question_bidirectional_attention_output], dim = 2)
        passage_lstm_and_bi_att_output_concatenated = torch.cat([passage_batch, passage_embedding, passage_bidirectional_attention_output], dim = 2)

        # each part of the tuples is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
        # input with the bi-att output
        passage_self_bi_att_output_lhs, passage_self_bi_att_output_rhs = self.passage_to_passage_attention(passage_embedding, passage_embedding)
        question_self_bi_att_output_lhs, question_self_bi_att_output_rhs = self.question_to_question_attention(question_embedding, question_embedding)

        # (N, length, 2 * INPUT_SIZE)
        passage_self_bi_att_output = (passage_self_bi_att_output_lhs + passage_self_bi_att_output_rhs)/2.0
        question_self_bi_att_output = (question_self_bi_att_output_lhs + question_self_bi_att_output_rhs)/2.0

        # (N, length, 2 * INPUT_SIZE) each
        question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
            self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)

        # (N, SEQUENCE_LENGTH, 6 * input_size)
        final_concatenated_passage_output = torch.cat([passage_lstm_and_bi_att_output_concatenated,
                                                       passage_bidirectional_attention_output_2], dim = 2)
        final_concatenated_question_output = torch.cat([question_lstm_and_bi_att_output_concatenated,
                                                        question_bidirectional_attention_output_2], dim = 2)

        # simply taking the final lstm state which should be (N, SEQUENCE_LENGTH , 2 * INPUT_SIZE)
        final_encoded_question_representation = self.final_question_encoder(final_concatenated_question_output,
                                              self.__init_lstm_hidden_and_cell_state(
                                                  final_concatenated_question_output.size()[0],
                                                  self.embedding.embedding_dim))
        final_question_encoding_raw = final_encoded_question_representation[1][0]
        final_question_encoding = torch.cat(torch.chunk(torch.transpose(final_question_encoding_raw, 0, 1), chunks = 2, dim = 1), dim = 2)
        final_question_encoding_scaled_to_passage_length = torch.cat([final_question_encoding] * final_concatenated_passage_output.shape[1], dim = 1)

        passage_outputs_enriched_with_final_question_encoding = torch.cat([final_concatenated_passage_output,
                                                                              final_question_encoding_scaled_to_passage_length], dim = 2)

        final_enriched_passage_output = self.lstm_for_final_question_and_passage_combination(passage_outputs_enriched_with_final_question_encoding,
                                                             self.__init_lstm_hidden_and_cell_state(
                                                                 passage_outputs_enriched_with_final_question_encoding.size()[0],
                                                                 self.embedding.embedding_dim * 2, is_bidirectional = False))[0]

        final_outputs_for_start_index = self.linear_layer_for_start_index(final_enriched_passage_output)
        final_outputs_for_end_index = self.linear_layer_for_end_index(final_enriched_passage_output)

        return final_outputs_for_start_index, final_outputs_for_end_index

    def __init_lstm_hidden_and_cell_state(self, batch_size: int, hidden_size: int, is_bidirectional: bool = True,
                                          num_layers: int = 1):
        return (
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device),
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device))


class QAModuleWithAttentionNoBertPassageOnly(Module):
    """
        this is kinda insane but is meant to detect bugs. The expectation is that this model performs
        very poorly
    """

    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__()
        self.embedding = embedding
        self.lstm_for_passage_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                       bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 2, 1, bias = False)
        self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 2, 1, bias = False)

        self.device = device

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):

        passage_batch = self.embedding(passage_batch_index_tensor)

        # the pytorch lstm outputs: output, (h_n, c_n)
        passage_lstm_output = self.lstm_for_passage_encoding(passage_batch, self.__init_lstm_hidden_and_cell_state(
            passage_batch.size()[0], self.embedding.embedding_dim))[0]

        final_outputs_for_start_index = self.linear_layer_for_start_index(passage_lstm_output)
        final_outputs_for_end_index = self.linear_layer_for_end_index(passage_lstm_output)

        return final_outputs_for_start_index, final_outputs_for_end_index

    def __init_lstm_hidden_and_cell_state(self, batch_size: int, hidden_size: int, is_bidirectional: bool = True,
                                          num_layers: int = 1):
        return (
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device),
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device))




