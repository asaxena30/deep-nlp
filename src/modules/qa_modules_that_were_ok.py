


# class QAModuleWithAttentionPointer(QAModule):
#
#     """adding a self-attention layer right before the output layer, partly inspired by pointer nets"""
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__(embedding = embedding, device = device)
#         self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 6)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
#                                                 bidirectional = True, batch_first = True, bias = False)
#
#         # using learned initial states for the embedding lstm
#         # self.lstm_for_embedding_init_hidden_state, self.lstm_for_embedding_init_cell_state = \
#         #     self._init_lstm_hidden_and_cell_state(1, self.embedding.embedding_dim, is_bidirectional = True, as_trainable_params = True)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim + self.embedding_lstm_hidden_size * 2, self.final_embedding_size)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module = BidirectionalAttention(self.final_embedding_size, activation = torch.nn.Tanh,
#                                                                      return_with_inputs_concatenated = True)
#
#         self.question_to_question_attention = SymmetricSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                      scale_attention_weights = True, activation = torch.nn.ReLU,
#                                                                      linear_layer_weight_init = kaiming_normal_)
#         self.passage_to_passage_attention = SymmetricSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                    scale_attention_weights = True, activation = torch.nn.ReLU,
#                                                                    linear_layer_weight_init = kaiming_normal_)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(self.final_embedding_size * 2,
#                                                                        return_with_inputs_concatenated = False, activation = torch.nn.Tanh)
#
#         self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 4, self.final_embedding_size * 2)
#         self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 2)
#
#         self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 2, self.final_embedding_size * 2,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 2 + 1, self.final_embedding_size * 2,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value = 1.0)
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value = 1.0)
#
#         self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
#             self._init_lstm_hidden_and_cell_state(1, self.final_embedding_size * 2, is_bidirectional = True, as_trainable_params = True)
#
#         self.final_self_attention_start_index = SymmetricSelfAttention(self.final_embedding_size * 4, return_with_inputs_concatenated = False,
#                                                                    scale_attention_weights = True, activation = torch.nn.Tanh,
#                                                                    linear_layer_weight_init = None)
#
#         self.final_self_attention_end_index = SymmetricSelfAttention(self.final_embedding_size * 4, return_with_inputs_concatenated = False,
#                                                                        scale_attention_weights = True, activation = torch.nn.Tanh,
#                                                                        linear_layer_weight_init = None)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # pdb.set_trace()
#
#         # using learned initial states for the embedding lstm
#         # embedding_init_states = (self.lstm_for_embedding_init_hidden_state.repeat(1, question_batch.size()[0], 1),
#         #  self.lstm_for_embedding_init_cell_state.repeat(1, question_batch.size()[0], 1))
#         #
#         # # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
#         # question_lstm_output = self.lstm_for_embedding(question_batch, embedding_init_states)[0]
#         #
#         # passage_lstm_output = self.lstm_for_embedding(passage_batch, embedding_init_states)[0]
#
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim = 2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output], dim = 2)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
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
#         final_concatenated_passage_output = torch.transpose(self.linear_layer_before_final_passage_rep(torch.cat([passage_bidirectional_attention_output,
#                                                                    passage_bidirectional_attention_output_2], dim = 2)), dim0 = 1, dim1 = 2)
#
#         final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(final_concatenated_passage_output), dim0 = 1, dim1 = 2)
#
#         init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#         init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#
#         # start index
#         final_enriched_passage_output_start_index = torch.cat(
#             [passage_embedding,
#              self.final_self_attention_start_index(
#                  self.lstm_for_final_passage_representation_start_index(
#                      final_concatenated_passage_output, (init_hidden_state_repeated, init_cell_state_repeated))[0])], dim = 2)
#
#         final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0 = 1, dim1 = 2)
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
#         final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))
#
#         # end index
#         final_enriched_passage_output_end_index = torch.cat(
#             [passage_embedding, self.final_self_attention_end_index(
#              self.lstm_for_final_passage_representation_end_index(
#                  torch.cat([final_concatenated_passage_output, final_outputs_for_start_index], dim = 2),
#                                                                                      (init_hidden_state_repeated,
#                                                                                       init_cell_state_repeated))[0])], dim = 2)
#
#         final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0 = 1, dim1 = 2)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))
#
#         return final_outputs_for_start_index, final_outputs_for_end_index




# class QAModuleWithAttentionModified(QAModule):
#
#     """ also uses self-attention before embedding"""
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__(embedding = embedding, device = device)
#         self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 6)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
#                                                 bidirectional = True, batch_first = True, bias = False)
#         self.question_to_question_attention_for_embedding = SymmetricSelfAttention(self.embedding.embedding_dim,
#                                                                                    return_with_inputs_concatenated = False,
#                                                                                    scale_attention_weights = True,
#                                                                                    activation = torch.nn.ReLU,
#                                                                                    linear_layer_weight_init = kaiming_normal_)
#         self.passage_to_passage_attention_for_embedding = SymmetricSelfAttention(self.embedding.embedding_dim,
#                                                                                  return_with_inputs_concatenated = False,
#                                                                                  scale_attention_weights = True,
#                                                                                  activation = torch.nn.ReLU,
#                                                                                  linear_layer_weight_init = kaiming_normal_)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim * 2 + self.embedding_lstm_hidden_size * 2, self.final_embedding_size)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module_1 = BidirectionalAttention(self.final_embedding_size, activation = torch.nn.Tanh,
#                                                                        return_with_inputs_concatenated = False)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(self.final_embedding_size,
#                                                                        return_with_inputs_concatenated = False, activation = torch.nn.Tanh)
#
#         self.question_to_question_attention = SymmetricSelfAttention(self.embedding.embedding_dim,
#                                                                      return_with_inputs_concatenated = False,
#                                                                      scale_attention_weights = True,
#                                                                      activation = torch.nn.ReLU,
#                                                                      linear_layer_weight_init = kaiming_normal_)
#
#         self.passage_to_passage_attention = SymmetricSelfAttention(self.embedding.embedding_dim,
#                                                                    return_with_inputs_concatenated = False,
#                                                                    scale_attention_weights = True,
#                                                                    activation = torch.nn.ReLU,
#                                                                    linear_layer_weight_init = kaiming_normal_)
#
#         self.bidirectional_attention_after_self_attention = BidirectionalAttention(self.final_embedding_size,
#                                                                        return_with_inputs_concatenated = False, activation = torch.nn.Tanh)
#
#         self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 5, self.final_embedding_size * 3)
#         self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 3)
#
#         output_lstms_hidden_size = int(self.final_embedding_size * 3 + self.final_embedding_size/3)
#         self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 3, output_lstms_hidden_size,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 3 + 1, output_lstms_hidden_size,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value = 1.0)
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value = 1.0)
#
#         self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
#             self._init_lstm_hidden_and_cell_state(1, output_lstms_hidden_size, is_bidirectional = True, as_trainable_params = True)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(output_lstms_hidden_size * 2)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(output_lstms_hidden_size * 2)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(output_lstms_hidden_size * 2, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(output_lstms_hidden_size * 2, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         # each is (N, length, INPUT_SIZE).
#         question_self_att_output_for_embedding = self.question_to_question_attention_for_embedding(question_batch)
#         passage_self_att_output_for_embedding = self.passage_to_passage_attention_for_embedding(passage_batch)
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output, passage_self_att_output_for_embedding], dim = 2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output, question_self_att_output_for_embedding], dim = 2)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#
#         # (N, seq_length,  2 * final_embedding_size)
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module_1(question_embedding, passage_embedding)
#
#         # (N, length, 2 * INPUT_SIZE) each
#         question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
#             self.bidirectional_attention_module_2(question_bidirectional_attention_output, passage_bidirectional_attention_output)
#
#         question_self_attention_output = self.question_to_question_attention(passage_embedding)
#         passage_self_attention_output = self.passage_to_passage_attention(passage_embedding)
#
#         passage_bidirectional_attention_output_3 = \
#             self.bidirectional_attention_after_self_attention(question_self_attention_output, passage_self_attention_output)[1]
#
#         # (N, SEQUENCE_LENGTH, 3 * input_size)
#         final_concatenated_passage_output =\
#             torch.transpose(self.linear_layer_before_final_passage_rep(torch.cat([passage_bidirectional_attention_output,
#                                                                                   passage_bidirectional_attention_output_2,
#                                                                                   passage_bidirectional_attention_output_3,
#                                                                                   passage_self_attention_output, passage_embedding], dim = 2)), dim0 = 1, dim1 = 2)
#
#         final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(final_concatenated_passage_output), dim0 = 1, dim1 = 2)
#
#         init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#         init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#
#         # start index
#         final_enriched_passage_output_start_index = self.lstm_for_final_passage_representation_start_index(final_concatenated_passage_output, (init_hidden_state_repeated,
#                                                                                                                            init_cell_state_repeated))[0]
#
#         final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0 = 1, dim1 = 2)
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
#         final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))
#
#         # end index
#         final_enriched_passage_output_end_index = self.lstm_for_final_passage_representation_end_index(torch.cat([final_concatenated_passage_output,
#                                                                                                 final_outputs_for_start_index], dim = 2),
#                                                                                      (init_hidden_state_repeated, init_cell_state_repeated))[0]
#
#         final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0 = 1, dim1 = 2)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))
#
#         return final_outputs_for_start_index, final_outputs_for_end_index






# class QAModuleWithGuidedSelfAttention(QAModule):
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__(embedding = embedding, device = device)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
#                                                 bidirectional = True, batch_first = True, bias = False)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.embedding.embedding_dim)
#
#         self.batch_norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.embedding.embedding_dim)
#
#         self.bidirectional_attention_module = BidirectionalAttention(self.embedding.embedding_dim, activation = torch.nn.Tanh)
#
#         self.lstm_for_question_encoding = torch.nn.LSTM(self.embedding.embedding_dim,
#                                                         self.embedding.embedding_dim,
#                                                         bidirectional = False, batch_first = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_question_encoding, value = 1.0)
#
#         self.question_to_question_attention = SymmetricSelfAttention(self.embedding.embedding_dim, return_with_inputs_concatenated = True,
#                                                                      scale_attention_weights = True, activation = torch.nn.ReLU,
#                                                                      linear_layer_weight_init = kaiming_normal_)
#         self.passage_to_passage_attention = GuidedSelfAttention(self.embedding.embedding_dim, return_with_inputs_concatenated = True,
#                                                                 scale_attention_weights = False, activation = torch.nn.ReLU,
#                                                                 linear_layer_weight_init = kaiming_normal_)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(self.embedding.embedding_dim * 2,
#                                                                        return_with_inputs_concatenated = False, activation = torch.nn.Tanh)
#
#         self.lstm_for_final_passage_representation = torch.nn.LSTM(self.embedding.embedding_dim * 4, self.embedding.embedding_dim * 2,
#                                                                    bidirectional = True, batch_first = True, bias = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation, value = 1.0)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.embedding.embedding_dim * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.embedding.embedding_dim * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
#         super.forward()
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding.embedding_dim))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
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
#         # (N, seq_length, 2 * embedding_size)
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module(question_embedding, passage_embedding)
#
#         question_encoding_for_guided_attention = self.lstm_for_question_encoding(question_embedding,
#                                                                                  self._init_lstm_hidden_and_cell_state(question_embedding.shape[0],
#                                                                                                                         question_embedding.shape[2],
#                                                                                                                         is_bidirectional = self.lstm_for_question_encoding.bidirectional))[
#             1][0]
#
#         question_encoding_for_guided_attention = torch.transpose(question_encoding_for_guided_attention, 0, 1)
#
#         # each is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
#         # input with the bi-att output
#         question_self_bi_att_output = self.question_to_question_attention(question_embedding)
#         passage_self_bi_att_output = self.passage_to_passage_attention(passage_embedding,
#                                                                        guidance_tensor = question_encoding_for_guided_attention)
#
#         # (N, length, 2 * INPUT_SIZE) each
#         question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
#             self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)
#
#         # (N, SEQUENCE_LENGTH, 4 * input_size)
#         final_concatenated_passage_output = torch.cat([passage_bidirectional_attention_output,
#                                                        passage_bidirectional_attention_output_2], dim = 2)
#
#         final_enriched_passage_output = torch.cat([passage_embedding, self.lstm_for_final_passage_representation(final_concatenated_passage_output,
#                                                                                                                  self._init_lstm_hidden_and_cell_state(
#                                                                                                                      final_concatenated_passage_output.size()[
#                                                                                                                          0],
#                                                                                                                      self.embedding.embedding_dim * 2,
#                                                                                                                      is_bidirectional = True))[0]],
#                                                   dim = 2)
#         # transpose for input to batch-norm layers
#         final_enriched_passage_output = torch.transpose(final_enriched_passage_output, dim0 = 1, dim1 = 2)
#
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output)
#
#         final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))
#         final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))
#
#         return final_outputs_for_start_index, final_outputs_for_end_index


# class QAModuleWithSimplifiedSelfAttention(QAModule):
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__(embedding = embedding, device = device)
#         self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 6)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
#                                                 bidirectional = True, batch_first = True, bias = False)
#
#         # using learned initial states for the embedding lstm
#         # self.lstm_for_embedding_init_hidden_state, self.lstm_for_embedding_init_cell_state = \
#         #     self._init_lstm_hidden_and_cell_state(1, self.embedding.embedding_dim, is_bidirectional = True, as_trainable_params = True)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim + self.embedding_lstm_hidden_size * 2, self.final_embedding_size)
#         xavier_normal_(self.linear_layer_for_final_embedding.weight)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module = BidirectionalAttention(self.final_embedding_size, activation = torch.nn.Tanh, scale_dot_products = True,
#                                                                      return_with_inputs_concatenated = True)
#
#         self.question_to_question_attention = SimplifiedSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                      scale_dot_products = True, activation = None)
#         self.passage_to_passage_attention = SimplifiedSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                    scale_dot_products = True, activation = None)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(self.final_embedding_size * 2,
#                                                                        return_with_inputs_concatenated = False, activation = torch.nn.Tanh,
#                                                                        scale_dot_products = True)
#
#         self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 4, self.final_embedding_size * 2)
#         self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 2)
#
#         xavier_normal_(self.linear_layer_before_final_passage_rep.weight)
#
#         self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 2, self.final_embedding_size * 2,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 2 + 1, self.final_embedding_size * 2,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value = 1.0)
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value = 1.0)
#
#         self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
#             self._init_lstm_hidden_and_cell_state(1, self.final_embedding_size * 2, is_bidirectional = True, as_trainable_params = True)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # pdb.set_trace()
#
#         # using learned initial states for the embedding lstm
#         # embedding_init_states = (self.lstm_for_embedding_init_hidden_state.repeat(1, question_batch.size()[0], 1),
#         #  self.lstm_for_embedding_init_cell_state.repeat(1, question_batch.size()[0], 1))
#         #
#         # # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
#         # question_lstm_output = self.lstm_for_embedding(question_batch, embedding_init_states)[0]
#         #
#         # passage_lstm_output = self.lstm_for_embedding(passage_batch, embedding_init_states)[0]
#
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim = 2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output], dim = 2)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
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
#         question_self_bi_att_output = (question_self_bi_att_output[0] + question_self_bi_att_output[1])/2
#         passage_self_bi_att_output = (passage_self_bi_att_output[0] + passage_self_bi_att_output[1])/2
#
#         # (N, length, 2 * INPUT_SIZE) each
#         question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
#             self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)
#
#         # (N, SEQUENCE_LENGTH, 2 * input_size)
#         final_concatenated_passage_output = torch.transpose(self.linear_layer_before_final_passage_rep(torch.cat([passage_bidirectional_attention_output,
#                                                                    passage_bidirectional_attention_output_2], dim = 2)), dim0 = 1, dim1 = 2)
#
#         final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(final_concatenated_passage_output), dim0 = 1, dim1 = 2)
#
#         init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#         init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#
#         # start index
#         final_enriched_passage_output_start_index = torch.cat(
#             [passage_embedding, self.lstm_for_final_passage_representation_start_index(final_concatenated_passage_output, (init_hidden_state_repeated,
#                                                                                                                            init_cell_state_repeated))[0]], dim = 2)
#
#         final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0 = 1, dim1 = 2)
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
#         final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))
#
#         # end index
#         final_enriched_passage_output_end_index = torch.cat(
#             [passage_embedding, self.lstm_for_final_passage_representation_end_index(torch.cat([final_concatenated_passage_output,
#                                                                                                 final_outputs_for_start_index], dim = 2),
#                                                                                      (init_hidden_state_repeated, init_cell_state_repeated))[0]], dim = 2)
#
#         final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0 = 1, dim1 = 2)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))
#
#         return final_outputs_for_start_index, final_outputs_for_end_index
#
# class QAModuleWithEnhancedSelfAttention(QAModule):
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__(embedding = embedding, device = device)
#         self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 6)
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
#                                                 bidirectional = True, batch_first = True, bias = False)
#
#         # using learned initial states for the embedding lstm
#         # self.lstm_for_embedding_init_hidden_state, self.lstm_for_embedding_init_cell_state = \
#         #     self._init_lstm_hidden_and_cell_state(1, self.embedding.embedding_dim, is_bidirectional = True, as_trainable_params = True)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim + self.embedding_lstm_hidden_size * 2, self.final_embedding_size)
#         xavier_normal_(self.linear_layer_for_final_embedding.weight)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module = BidirectionalAttention(self.final_embedding_size, activation = torch.nn.Tanh, scale_dot_products = True,
#                                                                      return_with_inputs_concatenated = True)
#
#         self.question_to_question_attention = EnhancedSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                      activation = torch.nn.CELU)
#
#         self.passage_to_passage_attention = EnhancedSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                      activation = torch.nn.CELU)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(self.final_embedding_size * 2,
#                                                                        return_with_inputs_concatenated = False, activation = torch.nn.Tanh,
#                                                                        scale_dot_products = True)
#
#         self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 4, self.final_embedding_size * 2)
#         self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 2)
#
#         xavier_normal_(self.linear_layer_before_final_passage_rep.weight)
#
#         self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 2, self.final_embedding_size * 2,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 2 + 1, self.final_embedding_size * 2,
#                                                                                bidirectional = True, batch_first = True, bias = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value = 1.0)
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value = 1.0)
#
#         self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
#             self._init_lstm_hidden_and_cell_state(1, self.final_embedding_size * 2, is_bidirectional = True, as_trainable_params = True)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # pdb.set_trace()
#
#         # using learned initial states for the embedding lstm
#         # embedding_init_states = (self.lstm_for_embedding_init_hidden_state.repeat(1, question_batch.size()[0], 1),
#         #  self.lstm_for_embedding_init_cell_state.repeat(1, question_batch.size()[0], 1))
#         #
#         # # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
#         # question_lstm_output = self.lstm_for_embedding(question_batch, embedding_init_states)[0]
#         #
#         # passage_lstm_output = self.lstm_for_embedding(passage_batch, embedding_init_states)[0]
#
#         question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]
#
#         passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim = 2)
#         question_embedding_input = torch.cat([question_batch, question_lstm_output], dim = 2)
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
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
#         final_concatenated_passage_output = torch.transpose(self.linear_layer_before_final_passage_rep(torch.cat([passage_bidirectional_attention_output,
#                                                                    passage_bidirectional_attention_output_2], dim = 2)), dim0 = 1, dim1 = 2)
#
#         final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(final_concatenated_passage_output), dim0 = 1, dim1 = 2)
#
#         init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#         init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
#
#         # start index
#         final_enriched_passage_output_start_index = torch.cat(
#             [passage_embedding, self.lstm_for_final_passage_representation_start_index(final_concatenated_passage_output, (init_hidden_state_repeated,
#                                                                                                                            init_cell_state_repeated))[0]], dim = 2)
#
#         final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0 = 1, dim1 = 2)
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
#         final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))
#
#         # end index
#         final_enriched_passage_output_end_index = torch.cat(
#             [passage_embedding, self.lstm_for_final_passage_representation_end_index(torch.cat([final_concatenated_passage_output,
#                                                                                                 final_outputs_for_start_index], dim = 2),
#                                                                                      (init_hidden_state_repeated, init_cell_state_repeated))[0]], dim = 2)
#
#         final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0 = 1, dim1 = 2)
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))
#
#         return final_outputs_for_start_index, final_outputs_for_end_index
