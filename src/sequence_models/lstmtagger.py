import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.standarddlutils import xavier_normal_weight_init


class LSTMTagger(nn.Module):
    def __init__(self, lstm_input_size: int, lstm_output_size: int, is_bidirectional: bool, num_layers: int, tagset_size: int):
        super(LSTMTagger, self).__init__()
        self.lstm_input_size = lstm_input_size
        self.lstm_output_size = lstm_output_size
        self.lstm = nn.LSTM(lstm_input_size, lstm_output_size, bidirectional=is_bidirectional, num_layers=num_layers,
                            dropout=0)
        xavier_normal_weight_init(self.lstm.named_parameters())
        self.num_directions = 2 if is_bidirectional else 1
        self.layer_norm = nn.LayerNorm(lstm_output_size * self.num_directions)
        self.output2TagLayer = nn.Linear(lstm_output_size * self.num_directions, tagset_size, bias=False)
        xavier_normal_weight_init(self.output2TagLayer.named_parameters())
        self.forward_iteration_counter: int = 0

    def __init_hidden_and_cell_state(self, batch_size: int):
        return (
            torch.zeros(self.lstm.num_layers * self.num_directions, batch_size, self.lstm_output_size),
            torch.zeros(self.lstm.num_layers * self.num_directions, batch_size, self.lstm_output_size))

    def forward(self, sentence_batch_embedding_tensor: torch.Tensor, batch_size: int):
        # print sentence_batch_embedding_tensor.shape
        lstm_out, final_hidden_and_cell_state = self.lstm(sentence_batch_embedding_tensor,
                                                          self.__init_hidden_and_cell_state(
                                                              batch_size))
        # print("lstm output shape:" + str(lstm_out.shape))

        output_tags_for_sentence_batches = self.output2TagLayer(lstm_out)  # num_words_in_sequence * batch_size * tagset_size
        softmax_output = F.softmax(output_tags_for_sentence_batches, dim=2)
        # if self.forward_iteration_counter % 1000 == 0:
        #     print_weights(self.lstm)
        self.forward_iteration_counter += 1
        return softmax_output
