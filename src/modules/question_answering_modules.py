from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertLayerNorm
from torch.nn.modules.module import Module
import torch.nn

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

        start_index_probabilities_tensor = self.linear_layer_for_answer_start_index(lstm_output[0])
        end_index_probabilities_tensor = self.linear_layer_for_answer_end_index(lstm_output[0])

        return start_index_probabilities_tensor, end_index_probabilities_tensor

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

        start_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_start_index(bert_model_output[0]))
        end_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_end_index(bert_model_output[0]))

        return start_index_probabilities_tensor, end_index_probabilities_tensor

    def __resolve_bert_model_size(self):
        return BERT_BASE_HIDDEN_SIZE if self.bert_model_name.startswith('bert-base') else BERT_LARGE_HIDDEN_SIZE
