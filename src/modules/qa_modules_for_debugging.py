import torch

from torch.nn import Module


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
