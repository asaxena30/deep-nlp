# class QAModuleWithAttentionTrimmed(Module):
#     """
#         Similar to it's older counterparts but has another skip connection right before the linear layer.
#
#         with gradient accumulation for 10 steps + Adam with initial-lr = 1e-3, batch-size  = 50 (not including the effect
#         of gradient accumulation as that'd effectively make it 500) + StepLR (step-wise decay of 0.1) + 2 epochs
#         (the interesting part is even though this leads to optimizer steps being reduced by a factor of 10, we get a much higher
#         EM accuracy. using accumulation for 5 steps leads to an EM accuracy around 27.7)
#
#         start index accuracy: 0.4542847503373819
#         end index accuracy: 0.4785762483130904
#         exact match accuracy: 0.33029689608636975
#         total answers: 5928
#
#         moving the batch-norm layer to after the linear embedding layer. best so far (using the above config):
#
#         start index accuracy: 0.4780701754385965
#         end index accuracy: 0.5015182186234818
#         exact match accuracy: 0.3535762483130904
#     """
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__()
#         self.embedding = embedding
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
#                                                 bidirectional = True, batch_first = True, bias = False)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.embedding.embedding_dim)
#         xavier_normal_(self.linear_layer_for_final_embedding.weight)
#
#         self.batch_norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.embedding.embedding_dim)
#
#         self.passage_to_passage_attention = SymmetricBidirectionalAttention(self.embedding.embedding_dim)
#         self.question_to_question_attention = SymmetricBidirectionalAttention(self.embedding.embedding_dim)
#
#         self.bidirectional_attention_module = BidirectionalAttention(self.embedding.embedding_dim * 2,
#                                                                      return_with_inputs_concatenated = False,
#                                                                      activation = torch.nn.Tanh, scale_dot_products = True)
#
#         self.lstm_for_final_passage_representation = torch.nn.LSTM(self.embedding.embedding_dim * 3, self.embedding.embedding_dim * 2,
#                                                                    bidirectional = True, batch_first = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation, value = 1.0)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.embedding.embedding_dim * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.embedding.embedding_dim * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1)
#
#         self.device = device
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
#         question_lstm_output = self.lstm_for_embedding(question_batch, self.__init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding.embedding_dim))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self.__init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding.embedding_dim))[0]
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim = 2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output], dim = 2)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.batch_norm_layer_for_embedding_output(
#             torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#         question_embedding = torch.transpose(self.batch_norm_layer_for_embedding_output(torch.transpose(
#             self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#
#         # each is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
#         # input with the bi-att output
#         passage_self_bi_att_output = self.passage_to_passage_attention(passage_embedding, passage_embedding)[0]
#         question_self_bi_att_output = self.question_to_question_attention(question_embedding, question_embedding)[0]
#
#         # (N, length, 2 * INPUT_SIZE) each
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module(question_self_bi_att_output, passage_self_bi_att_output)
#
#         # (N, SEQUENCE_LENGTH, 3 * input_size)
#         final_concatenated_passage_output = torch.cat([passage_embedding,
#                                                        passage_bidirectional_attention_output], dim = 2)
#
#         final_enriched_passage_output = torch.cat([passage_embedding, self.lstm_for_final_passage_representation(final_concatenated_passage_output,
#                                                                                                                  self.__init_lstm_hidden_and_cell_state(
#                                                                                                                      final_concatenated_passage_output.size()[
#                                                                                                                          0],
#                                                                                                                      self.embedding.embedding_dim * 2,
#                                                                                                                      is_bidirectional = True))[0]],
#                                                   dim = 2)
#
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(torch.transpose(final_enriched_passage_output, dim0 = 1, dim1 = 2))
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(torch.transpose(final_enriched_passage_output, dim0 = 1, dim1 = 2))
#
#         final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))
#         final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))
#
#         return final_outputs_for_start_index, final_outputs_for_end_index
#
#     def __init_lstm_hidden_and_cell_state(self, batch_size: int, hidden_size: int, is_bidirectional: bool = True,
#                                           num_layers: int = 1):
#         return (
#             torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
#                         device = self.device),
#             torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
#                         device = self.device))



# class QAModuleWithAttentionMod2(QAModule):
#     """
#     similar to QAModuleWithAttention but has exclusively 3 bi-attns. didn't work well initially as the accuracy went down to
#     39.2% from 42%+. But if the PReLU is tuned to change the param for the negative output form 0.25 to 0.15
#     (which brings it closer to ReLU) or of a GELU is used instead, the accuracy is 43%+ with the latter still slightly beating
#     the former (42.3% vs 42.05%)
#     """
#
#     def __init__(self, embedding: torch.nn.Embedding, device, summary_writer: SummaryWriter = None):
#         super().__init__(embedding=embedding, device=device, summary_writer=summary_writer)
#         self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 6)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
#                                                 bidirectional=True, batch_first=True, bias=False)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(
#             self.embedding.embedding_dim + self.embedding_lstm_hidden_size * 2,
#             self.final_embedding_size, bias=False)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module = BidirectionalAttention(input_size=self.final_embedding_size,
#                                                                      activation=torch.nn.Tanh,
#                                                                      return_with_inputs_concatenated=True,
#                                                                      scale_attention_weights_for_rhs=True,
#                                                                      scale_attention_weights_for_lhs=False)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(input_size=self.final_embedding_size,
#                                                                        return_with_inputs_concatenated=False,
#                                                                        activation=torch.nn.ReLU,
#                                                                        scale_attention_weights_for_rhs=True,
#                                                                        scale_attention_weights_for_lhs=False)
#
#         self.bidirectional_attention_module_3 = BidirectionalAttention(input_size=self.final_embedding_size,
#                                                                        return_with_inputs_concatenated=False,
#                                                                        activation=torch.nn.PReLU,
#                                                                        scale_attention_weights_for_rhs=True,
#                                                                        scale_attention_weights_for_lhs=False)
#
#         # trying something based on where we saw the training took the init param to last time
#         self.bidirectional_attention_module_3.activation = torch.nn.PReLU(init=0.15)
#
#         self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 4,
#                                                                      self.final_embedding_size * 2)
#
#         self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 2)
#
#         self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 2,
#                                                                                self.final_embedding_size * 2,
#                                                                                bidirectional=True, batch_first=True,
#                                                                                bias=True)
#
#         self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 2 + 1,
#                                                                              self.final_embedding_size * 2,
#                                                                              bidirectional=True, batch_first=True,
#                                                                              bias=True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value=1.0)
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value=1.0)
#
#         self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
#             self._init_lstm_hidden_and_cell_state(1, self.final_embedding_size * 2, is_bidirectional=True,
#                                                   as_trainable_params=True)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor,
#                 iteration_num: int = None):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # pdb.set_trace()
#
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim=2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output], dim=2)
#
#         passage_embedding_input = self.linear_layer_for_final_embedding(passage_embedding_input)
#         question_embedding_input = self.linear_layer_for_final_embedding(question_embedding_input)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(passage_embedding_input, dim0=1, dim1=2)), dim0=1, dim1=2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             question_embedding_input, dim0=1, dim1=2)), dim0=1, dim1=2)
#
#         # (N, seq_length,  2 * embedding_size)
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module(question_embedding, passage_embedding)
#
#         # (N, length, 2 * INPUT_SIZE) each since the input is 2 * INPUT_SIZE
#         question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
#             self.bidirectional_attention_module_2(question_embedding, passage_embedding)
#
#         question_bidirectional_attention_output_3, passage_bidirectional_attention_output_3 = \
#             self.bidirectional_attention_module_3(question_embedding, passage_embedding)
#
#         # (N, SEQUENCE_LENGTH, 2 * input_size)
#         final_concatenated_passage_output = self.linear_layer_before_final_passage_rep(
#             torch.cat([passage_bidirectional_attention_output,
#                        passage_bidirectional_attention_output_2,
#                        passage_bidirectional_attention_output_3], dim=2))
#
#         final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(
#             torch.transpose(final_concatenated_passage_output, dim0=1, dim1=2)), dim0=1, dim1=2)
#
#         init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1,
#                                                                                               final_concatenated_passage_output.size()[
#                                                                                                   0], 1)
#         init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1,
#                                                                                           final_concatenated_passage_output.size()[
#                                                                                               0], 1)
#
#         # start index
#         final_enriched_passage_output_start_index = torch.cat(
#             [passage_embedding,
#              self.lstm_for_final_passage_representation_start_index(final_concatenated_passage_output,
#                                                                     (init_hidden_state_repeated,
#                                                                      init_cell_state_repeated))[0]], dim=2)
#
#         final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0=1,
#                                                                     dim1=2)
#
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
#         final_outputs_for_start_index = self.linear_layer_for_start_index(
#             torch.transpose(batch_normed_start_index_output, dim0=1, dim1=2))
#
#         # end index
#         final_enriched_passage_output_end_index = torch.cat(
#             [passage_embedding,
#              self.lstm_for_final_passage_representation_end_index(torch.cat([final_concatenated_passage_output,
#                                                                              final_outputs_for_start_index], dim=2),
#                                                                   (init_hidden_state_repeated,
#                                                                    init_cell_state_repeated))[0]], dim=2)
#
#         final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0=1,
#                                                                   dim1=2)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(
#             torch.transpose(batch_normed_end_index_output, dim0=1, dim1=2))
#
#         if self.training and False:
#             self.log_tensor_stats({"passage/embedding_input": passage_embedding_input,
#                                    "question/embedding_input": question_embedding_input,
#                                    "passage/embedding": passage_embedding, "question/embedding": question_embedding,
#                                    "passage/bi_attn": passage_bidirectional_attention_output,
#                                    "question/bi_attn": question_bidirectional_attention_output,
#                                    "passage/bi_attn_2": passage_bidirectional_attention_output_2,
#                                    "passage/bi_attn_3": passage_bidirectional_attention_output_3,
#                                    "passage/final_enriched_output_start_idx": final_enriched_passage_output_start_index,
#                                    "passage/batch_normed_start_index_output": batch_normed_start_index_output,
#                                    "passage/final_outputs_for_start_index": final_outputs_for_start_index,
#                                    "passage/final_enriched_passage_output_end_index": final_enriched_passage_output_end_index,
#                                    "passage/batch_normed_end_index_output": batch_normed_end_index_output,
#                                    "passage/final_outputs_for_end_index": final_outputs_for_end_index
#                                    }, iteration_num)
#
#         return final_outputs_for_start_index, final_outputs_for_end_index


# class QAModuleWithAttentionMod(QAModule):
#     """
#     similar to QAModuleWithAttention but only has 2 bi-attns and no self-attn. The input of one is being fed to the other.
#     Accuracy decreased to 40.4% from around 42%+ for the original model
#     """
#
#     def __init__(self, embedding: torch.nn.Embedding, device, summary_writer: SummaryWriter = None):
#         super().__init__(embedding=embedding, device=device, summary_writer=summary_writer)
#         self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 6)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
#                                                 bidirectional=True, batch_first=True, bias=False)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(
#             self.embedding.embedding_dim + self.embedding_lstm_hidden_size * 2,
#             self.final_embedding_size, bias=False)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module = BidirectionalAttention(input_size=self.final_embedding_size,
#                                                                      activation=torch.nn.Tanh,
#                                                                      return_with_inputs_concatenated=True,
#                                                                      scale_attention_weights_for_rhs=True,
#                                                                      scale_attention_weights_for_lhs=False)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(input_size=self.final_embedding_size * 2,
#                                                                        return_with_inputs_concatenated=False,
#                                                                        activation=torch.nn.Tanh,
#                                                                        scale_attention_weights_for_rhs=True,
#                                                                        scale_attention_weights_for_lhs=False)
#
#         self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 4,
#                                                                      self.final_embedding_size * 2)
#
#         self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 2)
#
#         self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 2,
#                                                                                self.final_embedding_size * 2,
#                                                                                bidirectional=True, batch_first=True,
#                                                                                bias=True)
#
#         self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 2 + 1,
#                                                                              self.final_embedding_size * 2,
#                                                                              bidirectional=True, batch_first=True,
#                                                                              bias=True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value=1.0)
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value=1.0)
#
#         self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
#             self._init_lstm_hidden_and_cell_state(1, self.final_embedding_size * 2, is_bidirectional=True,
#                                                   as_trainable_params=True)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor,
#                 iteration_num: int = None):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # pdb.set_trace()
#
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim=2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output], dim=2)
#
#         passage_embedding_input = self.linear_layer_for_final_embedding(passage_embedding_input)
#         question_embedding_input = self.linear_layer_for_final_embedding(question_embedding_input)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(passage_embedding_input, dim0=1, dim1=2)), dim0=1, dim1=2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             question_embedding_input, dim0=1, dim1=2)), dim0=1, dim1=2)
#
#         # (N, seq_length,  2 * embedding_size)
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module(question_embedding, passage_embedding)
#
#         # (N, length, 2 * embedding_size) each since the input is 2 * embedding_size
#         question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
#             self.bidirectional_attention_module_2(question_bidirectional_attention_output,
#                                                   passage_bidirectional_attention_output)
#
#         # (N, SEQUENCE_LENGTH, 2 * input_size)
#         final_concatenated_passage_output = self.linear_layer_before_final_passage_rep(
#             torch.cat([passage_bidirectional_attention_output,
#                        passage_bidirectional_attention_output_2], dim=2))
#
#         final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(
#             torch.transpose(final_concatenated_passage_output, dim0=1, dim1=2)), dim0=1, dim1=2)
#
#         init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1,
#                                                                                               final_concatenated_passage_output.size()[
#                                                                                                   0], 1)
#         init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1,
#                                                                                           final_concatenated_passage_output.size()[
#                                                                                               0], 1)
#
#         # start index
#         final_enriched_passage_output_start_index = torch.cat(
#             [passage_embedding,
#              self.lstm_for_final_passage_representation_start_index(final_concatenated_passage_output,
#                                                                     (init_hidden_state_repeated,
#                                                                      init_cell_state_repeated))[0]], dim=2)
#
#         final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0=1,
#                                                                     dim1=2)
#
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
#         final_outputs_for_start_index = self.linear_layer_for_start_index(
#             torch.transpose(batch_normed_start_index_output, dim0=1, dim1=2))
#
#         # end index
#         final_enriched_passage_output_end_index = torch.cat(
#             [passage_embedding,
#              self.lstm_for_final_passage_representation_end_index(torch.cat([final_concatenated_passage_output,
#                                                                              final_outputs_for_start_index], dim=2),
#                                                                   (init_hidden_state_repeated,
#                                                                    init_cell_state_repeated))[0]], dim=2)
#
#         final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0=1,
#                                                                   dim1=2)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(
#             torch.transpose(batch_normed_end_index_output, dim0=1, dim1=2))
#
#         if self.training and False:
#             self.log_tensor_stats({"passage/embedding_input": passage_embedding_input,
#                                    "question/embedding_input": question_embedding_input,
#                                    "passage/embedding": passage_embedding, "question/embedding": question_embedding,
#                                    "passage/bi_attn": passage_bidirectional_attention_output,
#                                    "question/bi_attn": question_bidirectional_attention_output,
#                                    "passage/bi_attn_2": passage_bidirectional_attention_output_2,
#                                    "passage/final_enriched_output_start_idx": final_enriched_passage_output_start_index,
#                                    "passage/batch_normed_start_index_output": batch_normed_start_index_output,
#                                    "passage/final_outputs_for_start_index": final_outputs_for_start_index,
#                                    "passage/final_enriched_passage_output_end_index": final_enriched_passage_output_end_index,
#                                    "passage/batch_normed_end_index_output": batch_normed_end_index_output,
#                                    "passage/final_outputs_for_end_index": final_outputs_for_end_index
#                                    }, iteration_num)
#
#         return final_outputs_for_start_index, final_outputs_for_end_index




# class QAModuleWithAttentionMod(QAModule):
#     """
#     using modified self-attention which puts the linear layer after the attention calc as oppsoed to applying it to
#     attention inputs, accuracy went down to 39.6%
#
#     """
#
#     def __init__(self, embedding: torch.nn.Embedding, device, summary_writer: SummaryWriter = None):
#         super().__init__(embedding=embedding, device=device, summary_writer=summary_writer)
#         self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 6)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
#                                                 bidirectional=True, batch_first=True, bias=False)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(
#             self.embedding.embedding_dim + self.embedding_lstm_hidden_size * 2,
#             self.final_embedding_size, bias=False)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module = BidirectionalAttention(input_size=self.final_embedding_size,
#                                                                      activation=torch.nn.Tanh,
#                                                                      return_with_inputs_concatenated=True,
#                                                                      scale_attention_weights_for_rhs=True,
#                                                                      scale_attention_weights_for_lhs=False)
#
#         self.question_to_question_attention = ModifiedSymmetricSelfAttention(input_size=self.final_embedding_size,
#                                                                              return_with_inputs_concatenated=True,
#                                                                              scale_attention_weights=True,
#                                                                              activation=torch.nn.ReLU,
#                                                                              linear_layer_weight_init=kaiming_normal_)
#
#         self.passage_to_passage_attention = ModifiedSymmetricSelfAttention(input_size=self.final_embedding_size,
#                                                                            return_with_inputs_concatenated=True,
#                                                                            scale_attention_weights=True,
#                                                                            activation=torch.nn.ReLU,
#                                                                            linear_layer_weight_init=kaiming_normal_)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(input_size=self.final_embedding_size * 2,
#                                                                        return_with_inputs_concatenated=False,
#                                                                        activation=torch.nn.Tanh,
#                                                                        scale_attention_weights_for_rhs=True,
#                                                                        scale_attention_weights_for_lhs=False)
#
#         self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 4,
#                                                                      self.final_embedding_size * 2)
#
#         self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 2)
#
#         self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 2,
#                                                                                self.final_embedding_size * 2,
#                                                                                bidirectional=True, batch_first=True,
#                                                                                bias=True)
#
#         self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 2 + 1,
#                                                                              self.final_embedding_size * 2,
#                                                                              bidirectional=True, batch_first=True,
#                                                                              bias=True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value=1.0)
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value=1.0)
#
#         self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
#             self._init_lstm_hidden_and_cell_state(1, self.final_embedding_size * 2, is_bidirectional=True,
#                                                   as_trainable_params=True)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor,
#                 iteration_num: int = None):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # pdb.set_trace()
#
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim=2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output], dim=2)
#
#         passage_embedding_input = self.linear_layer_for_final_embedding(passage_embedding_input)
#         question_embedding_input = self.linear_layer_for_final_embedding(question_embedding_input)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(passage_embedding_input, dim0=1, dim1=2)), dim0=1, dim1=2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             question_embedding_input, dim0=1, dim1=2)), dim0=1, dim1=2)
#
#         # (N, seq_length,  2 * embedding_size)
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module(question_embedding, passage_embedding)
#
#         # each is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
#         # input with the bi-att output
#         question_self_bi_att_output = self.question_to_question_attention(question_embedding)
#         passage_self_bi_att_output = self.passage_to_passage_attention(passage_embedding)
#
#         # (N, length, 2 * INPUT_SIZE) each
#         question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
#             self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)
#
#         # (N, SEQUENCE_LENGTH, 2 * input_size)
#         final_concatenated_passage_output = self.linear_layer_before_final_passage_rep(
#             torch.cat([passage_bidirectional_attention_output,
#                        passage_bidirectional_attention_output_2], dim=2))
#
#         final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(
#             torch.transpose(final_concatenated_passage_output, dim0=1, dim1=2)), dim0=1, dim1=2)
#
#         init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1,
#                                                                                               final_concatenated_passage_output.size()[
#                                                                                                   0], 1)
#         init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1,
#                                                                                           final_concatenated_passage_output.size()[
#                                                                                               0], 1)
#
#         # start index
#         final_enriched_passage_output_start_index = torch.cat(
#             [passage_embedding,
#              self.lstm_for_final_passage_representation_start_index(final_concatenated_passage_output,
#                                                                     (init_hidden_state_repeated,
#                                                                      init_cell_state_repeated))[0]], dim=2)
#
#         final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0=1,
#                                                                     dim1=2)
#
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
#         final_outputs_for_start_index = self.linear_layer_for_start_index(
#             torch.transpose(batch_normed_start_index_output, dim0=1, dim1=2))
#
#         # end index
#         final_enriched_passage_output_end_index = torch.cat(
#             [passage_embedding,
#              self.lstm_for_final_passage_representation_end_index(torch.cat([final_concatenated_passage_output,
#                                                                              final_outputs_for_start_index], dim=2),
#                                                                   (init_hidden_state_repeated,
#                                                                    init_cell_state_repeated))[0]], dim=2)
#
#         final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0=1,
#                                                                   dim1=2)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(
#             torch.transpose(batch_normed_end_index_output, dim0=1, dim1=2))
#
#         if self.training and False:
#             self.log_tensor_stats({"passage/embedding_input": passage_embedding_input,
#                                    "question/embedding_input": question_embedding_input,
#                                    "passage/embedding": passage_embedding, "question/embedding": question_embedding,
#                                    "passage/bi_attn": passage_bidirectional_attention_output,
#                                    "question/bi_attn": question_bidirectional_attention_output,
#                                    "passage/self_attn": passage_self_bi_att_output,
#                                    "question/self_attn": question_self_bi_att_output,
#                                    "passage/bi_attn_2": passage_bidirectional_attention_output_2,
#                                    "passage/final_enriched_output_start_idx": final_enriched_passage_output_start_index,
#                                    "passage/batch_normed_start_index_output": batch_normed_start_index_output,
#                                    "passage/final_outputs_for_start_index": final_outputs_for_start_index,
#                                    "passage/final_enriched_passage_output_end_index": final_enriched_passage_output_end_index,
#                                    "passage/batch_normed_end_index_output": batch_normed_end_index_output,
#                                    "passage/final_outputs_for_end_index": final_outputs_for_end_index
#                                    }, iteration_num)
#
#         return final_outputs_for_start_index, final_outputs_for_end_index
