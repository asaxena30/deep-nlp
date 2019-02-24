from pytorch_pretrained_bert import BertModel
from torch.nn.modules.module import Module
import torch.nn

BERT_BASE_HIDDEN_SIZE: int = 768
BERT_LARGE_HIDDEN_SIZE: int = 1024


class BertQuestionAnsweringModule(Module):

    def __init__(self, bert_model_name = 'bert-base-uncased', batch_first = True, device = None,
                 named_param_weight_initializer = None):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
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

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False):
        bert_model_output: torch.Tensor = self.bert_model(input_ids = input_ids, token_type_ids = token_type_ids,
                                                          attention_mask = attention_mask,
                                                          output_all_encoded_layers = output_all_encoded_layers)
        lstm_output = self.lstm(bert_model_output[0], self.__init_lstm_hidden_and_cell_state(
                                                              input_ids.size()[0]))
        start_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_start_index(lstm_output[0]))
        end_index_probabilities_tensor = self.softmax(self.linear_layer_for_answer_end_index(lstm_output[0]))

        return start_index_probabilities_tensor, end_index_probabilities_tensor

    def __resolve_bert_model_size(self):
        return BERT_BASE_HIDDEN_SIZE if self.bert_model_name.startswith('bert-base') else BERT_LARGE_HIDDEN_SIZE

    def __init_lstm_hidden_and_cell_state(self, batch_size: int):
        return (
            torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.lstm.hidden_size,
                        device = self.device),
            torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.lstm.hidden_size,
                        device = self.device))

