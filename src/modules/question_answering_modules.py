import torch
import torch.nn
from torch.nn.init import xavier_normal_, kaiming_normal_
from torch.nn.modules.module import Module
import pdb

from src.common.neural_net_param_utils import init_lstm_forget_gate_biases
from src.modules.attention.attention_modules import \
    SymmetricBidirectionalAttention, BidirectionalAttention, GuidedSelfAttention, SymmetricSelfAttention, \
    BidafAttention, AsymmetricSelfAttention, BidirectionalAttentionSimplified


class QAModule(Module):

    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__()
        self.embedding = embedding
        self.device = device

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
        pass

    def _init_lstm_hidden_and_cell_state(self, batch_size: int, hidden_size: int, is_bidirectional: bool = True,
                                         num_layers: int = 1, as_trainable_params = False):
        return (
            torch.nn.Parameter(torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device)),
            torch.nn.Parameter(torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device))) if as_trainable_params else (
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device),
            torch.zeros(num_layers * (2 if is_bidirectional else 1), batch_size, hidden_size,
                        device = self.device))


class QAModuleWithGuidedSelfAttention(QAModule):

    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__(embedding = embedding, device = device)
        self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
                                                bidirectional = True, batch_first = True, bias = False)

        self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.embedding.embedding_dim)
        xavier_normal_(self.linear_layer_for_final_embedding.weight)

        self.batch_norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.embedding.embedding_dim)

        self.bidirectional_attention_module = BidirectionalAttention(self.embedding.embedding_dim, activation = torch.nn.Tanh,
                                                                     scale_dot_products = True)

        self.lstm_for_question_encoding = torch.nn.LSTM(self.embedding.embedding_dim,
                                                        self.embedding.embedding_dim,
                                                        bidirectional = False, batch_first = True)

        init_lstm_forget_gate_biases(self.lstm_for_question_encoding, value = 1.0)

        self.question_to_question_attention = SymmetricSelfAttention(self.embedding.embedding_dim, return_with_inputs_concatenated = True,
                                                                     scale_dot_products = True, activation = torch.nn.ReLU,
                                                                     linear_layer_weight_init = kaiming_normal_)
        self.passage_to_passage_attention = GuidedSelfAttention(self.embedding.embedding_dim, return_with_inputs_concatenated = True,
                                                                scale_dot_products = False, activation = torch.nn.ReLU,
                                                                linear_layer_weight_init = kaiming_normal_)

        self.bidirectional_attention_module_2 = BidirectionalAttention(self.embedding.embedding_dim * 2,
                                                                       return_with_inputs_concatenated = False, activation = torch.nn.Tanh,
                                                                       scale_dot_products = True)

        self.lstm_for_final_passage_representation = torch.nn.LSTM(self.embedding.embedding_dim * 4, self.embedding.embedding_dim * 2,
                                                                   bidirectional = True, batch_first = True, bias = True)

        init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation, value = 1.0)

        self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.embedding.embedding_dim * 5)
        self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.embedding.embedding_dim * 5)

        self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1)
        self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1)

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
        question_batch = self.embedding(question_batch_index_tensor)
        passage_batch = self.embedding(passage_batch_index_tensor)

        # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
        question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
            question_batch.size()[0], self.embedding.embedding_dim))[0]

        passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
            passage_batch.size()[0], self.embedding.embedding_dim))[0]

        passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim = 2)
        question_embedding_input = torch.cat([question_batch, question_lstm_output], dim = 2)

        # (N, seq_length, embedding_size)
        passage_embedding = torch.transpose(self.batch_norm_layer_for_embedding_output(
            torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
        question_embedding = torch.transpose(self.batch_norm_layer_for_embedding_output(torch.transpose(
            self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)

        # (N, seq_length, 2 * embedding_size)
        question_bidirectional_attention_output, passage_bidirectional_attention_output = \
            self.bidirectional_attention_module(question_embedding, passage_embedding)

        question_encoding_for_guided_attention = self.lstm_for_question_encoding(question_embedding,
                                                                                 self._init_lstm_hidden_and_cell_state(question_embedding.shape[0],
                                                                                                                        question_embedding.shape[2],
                                                                                                                        is_bidirectional = self.lstm_for_question_encoding.bidirectional))[
            1][0]

        question_encoding_for_guided_attention = torch.transpose(question_encoding_for_guided_attention, 0, 1)

        # each is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
        # input with the bi-att output
        question_self_bi_att_output = self.question_to_question_attention(question_embedding)
        passage_self_bi_att_output = self.passage_to_passage_attention(passage_embedding,
                                                                       guidance_tensor = question_encoding_for_guided_attention)

        # (N, length, 2 * INPUT_SIZE) each
        question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
            self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)

        # (N, SEQUENCE_LENGTH, 4 * input_size)
        final_concatenated_passage_output = torch.cat([passage_bidirectional_attention_output,
                                                       passage_bidirectional_attention_output_2], dim = 2)

        final_enriched_passage_output = torch.cat([passage_embedding, self.lstm_for_final_passage_representation(final_concatenated_passage_output,
                                                                                                                 self._init_lstm_hidden_and_cell_state(
                                                                                                                     final_concatenated_passage_output.size()[
                                                                                                                         0],
                                                                                                                     self.embedding.embedding_dim * 2,
                                                                                                                     is_bidirectional = True))[0]],
                                                  dim = 2)
        # transpose for input to batch-norm layers
        final_enriched_passage_output = torch.transpose(final_enriched_passage_output, dim0 = 1, dim1 = 2)

        batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output)
        batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output)

        final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))
        final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))

        return final_outputs_for_start_index, final_outputs_for_end_index



class QAModuleWithAttentionLarge(QAModule):
    """
        Typical EM accuracy as of #16 - 17 below seems to be 44%+ with 2 epochs, batch-size = 32, grad-accum = 16 steps, Adam with lr = 1e-3 with step-decay
        of 0.1 per epoch. training for 2 epochs

        Observations (2 epoch training):
         1. Sometimes making the model wider is taking a small toll at accuracy. for eg. using a separate bi-attn #2 layer
            for start vs end index results in a lower accuracy by ~ 1%. This could be because we're only training for 2 epochs
         2. Similar to #2, making the final linear layer deal with an input of embedding-size * 9 (obtained by not having the final passage rep lstm
            squeeze the input to half it's size), the accuracy went down by ~3%. the hypothesis is that the final single linear layer has trouble dealing
            with the noise generated by a very large input-size. This might potentially be cured by training for more than 2 epochs (current)
         3. Using asymmetric self-attention as opposed to symmetric everywhere didn't seem to have an accuracy gain. Again this might change when training for more than 2 epochs
         4. Using BertAdam with warmup = 0.1 didn't seem to provide any accuracy gain (rest everything is same including training time of 2 epochs). The only change
            was, gradient-clipping (the default for bertadam) was disabled
         5. 09/25/2019 - for this 2 epoch training, using guided self-attention (using question encoding as a gate when multiplying passage word components)
            didn't yield any gain in accuracy. This was done through a separate model (called QAModuleWithGuidedSelfAttention at the time of this writing).
            The latter was identical to this model except for the use of guided self-attention for passage <-> passage attention
         6. 09/26/2019: trying learnable initial hidden/cell states for final passage rep lstm. no gain in accuracy found for 2 epoch training, test accuracy still lurking
            around ~37%
         7. Added hierarchical attention (passage-self-attention-output X question-embedding concatenated with question-bi-attn-output X passage embedding).
            Accuracy went down by 5% to ~ 32% for 2 epoch training. Squeezing in another batch-norm makes accuracy go up tp 35% in spite of this new attention

            Getting rid of #7's change from here onwards

         8. Feeding the start index outputs while calculating the end index outputs made the EM accuracy go up to 40.98%
         9. Using learned hidden states for the embedding lstm actually decreased accuracy back to ~37% in spite of retaining #8's changes
         10. Using two linear layers at the end with the 1st linear layer squeezing the outputs to embedding_size * 3 (which is when input to the final linear layer) + relu
            after the first linear layer + #8 gave the same accuracy as #8 ~ 40.99%
         11. Adding another bi-attention layer on top of #10 squeezes the accuracy down to 39.37%. this bi-attn layer computes
         another bi-attention on top of 1st bi-attn outputs
         12. Providing a bigger intermediate layer or a bigger 2nd linear output layer decreases performance to around 37%
         13. Going back to #10 (throwing away the bigger intermediate layer in #12 and the 3rd bi-attn layer in #11) but using glove.800b.300d (while only keeping the words/tokens required for squad train/dev datasets) as opposed to the
             available FastText (which isn't the same as FastText 800B) increases the accuracy to 43.91% => embedding quality matters.
             Most likely glove 6B etc. won't be able to reproduce the same results. FastText's authors, in their paper claimed their embedding to be superior to glove's.
             There's a chance that upon using FastText 800B etc., we'd see similar/better improvements
             On a side note, limiting the embedding matrix to only the tokens required for squad, drastically reduces the memory footprint (65 - 70%) -> ~40%
             of V100 memory
         14. Trying two separate LSTMs for the start and end index respectively. this has been tried before with no advantage but the difference this time is, the
             end-index lstm is fed the start-index final output. Accuracy was similar to #13 => 43.82%. The difference seen was, now the start and end index individual accuracies
             are closer to each other 56.x vs 57.y % compared to #13 (~54% and ~58% respectively). Memory usage on V100 went up from ~ 40% to 48.8% compared
             to #13 due to the addition of the new lstm
         15. Same as #14 but using a detach() on the start-index input makes the accuracy go down to 43.42%. The effect detach has here is that the end-index sub-graph training won't
             lead to a change in params for the entire start-index subgraph thus in a way favoring that start-index affect the end-index but the training dedicated to the latter doesn't
             affect the former. Looks like at least for the first 2 epochs, removing this co-adaptation decreases accuracy even if by 0.4 %
         16. Same as #14 (no longer using detach() as noted in 315 above) but initializing unknown word embeddings to torch.randn(300)/100. the accuracy increases to 44.3%.
             When using torch.randn(300)/10 instead, the accuracy goes down to around 43.3% (which is lower than simply initializing
             each unknown word to it's individual 0-tensor)
         17. Adding two embedding lstms with a skip connection between them (2nd lstm input = 1st lstm output + original passage/question batch), accuracy is mostly
             the same as #16 (44.2% as opposed to 44.3% in #16)
         18. Using a leaky-ReLU for the first bidirectional-attention (as opposed to Tanh) led to an accuracy of 43.43. It's comparable to using a Tanh since for one of the runs
             tanh yielded ~43% accuracy
         19. Using bidaf-attention for a 2 epoch training decreased the accuracy to ~40%
         20. Removed bidaf attention and leaky-relu (back to tanh). In the bidrectional attention, it was found that the attention weights for LHS (question) were turning out to be very
             similar for each passage word i.e. the attention paid by each question word to each passage was very homogeneous across passage words. Taking away the
             denominator of sq-root of embedding-size from final_attention_weights_softmaxed_for_lhs made accuracy increase to 44.9%
         21. Using simplified bi-attention (where the lhs and rhs are only multiplied by vectors initialized to 1s as opposed to being passed through linear layers),
             the accuracy observed was 44.3%. For this run, I retained the change where only the rhs aka question-weights for each context word were being divided by
             sq-root of input-size
         22. Using unnormalized weights (i.e. non-softmaxed) even for simplified biattention reduces the accuracy to 40.35%. "even" because the same effect
             was observed for the case of standard bi-attention
         23. Same as #20 (which led to 44.9% accuracy) but using RAdam with lr = 1e-2 (as opposed to be 1e-3 for most other runs that used Adam, including #20). Interestingly,
             using 1e-3 with RAdam led to very high errors (0.3+) even until iteration #4000+ (stopped this run before it completed after seeing this).
             Also, the avg. loss was close (in fact lower at the 4th place of decimal) to #20. 0.2363 for #20 vs 0.2360 for RAdam. Looks like RAdam
             will allow a higher learning rate as reported in online literature/discussion-threads. This might prove useful for > 2 epoch training.
             Verified again that using 1e-2 with Adam is disastrous as expected from very early runs (loss = 0.48 around iteration# 3900).
             This was worse than using 1e-3 with RAdam
         24. On a slightly different note, tried AdamW as well. first with weight-decay = 1e-2 (this run was using the simplified bi-attention) => accuracy was
             around 40%. However, when beefing up the regularization to 5e-2 (and switching to #20 architecture i.e. going back to standard bi-attention),
             the dev accuracy went down to 27% => let's avoid using too high of a weight decay right now at least for the 2 epoch training
         25. Using a larger hidden state on the embedding lstm (350 as opposed to 300) increases EM accuracy to 44.94%. This also involved a bug-fix
             which was caused by pytorch’s new BoolTensor behavior which was making the accuracy seem like 56.68% which was identical to the start-index accuracy.
             Using a hidden state of 400 decreases the accuracy to 43.7%


            TODO: make batch-size a constructor param?

            1. Layer-wise learning rates: https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
            2. evaluate saved models: https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/32

        Conneau et al (https://arxiv.org/pdf/1705.02364.pdf) had an interesting observation that while sentence representation models trained using attention
        can fit faster, LSTM models could generalize better to other tasks.

        In their case a BI-LSTM with max-pooling between the last forward and backward hidden states yielded the best sentence rep.
        The caveat is, these models have been trained on SNLI and possibly for a lot of epochs. This paper doesn't mention the
        exact number of epochs, simply that they conditionally decrease the lr whenever the dev accuracy after an epoch goes down and stop training once the lr
        goes below 1e-5

    """

    def __init__(self, embedding: torch.nn.Embedding, device):
        super().__init__(embedding = embedding, device = device)
        self.embedding_lstm_hidden_size = int(self.embedding.embedding_dim + self.embedding.embedding_dim / 3)
        self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding_lstm_hidden_size,
                                                bidirectional = True, batch_first = True, bias = False)

        # using learned initial states for the embedding lstm
        # self.lstm_for_embedding_init_hidden_state, self.lstm_for_embedding_init_cell_state = \
        #     self._init_lstm_hidden_and_cell_state(1, self.embedding.embedding_dim, is_bidirectional = True, as_trainable_params = True)

        self.final_embedding_size = int(self.embedding.embedding_dim)

        self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim + self.embedding_lstm_hidden_size * 2, self.final_embedding_size)
        xavier_normal_(self.linear_layer_for_final_embedding.weight)

        self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)

        self.bidirectional_attention_module = BidirectionalAttention(self.final_embedding_size, activation = torch.nn.Tanh, scale_dot_products = True,
                                                                     return_with_inputs_concatenated = True)

        self.question_to_question_attention = SymmetricSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
                                                                     scale_dot_products = True, activation = torch.nn.ReLU,
                                                                     linear_layer_weight_init = kaiming_normal_)
        self.passage_to_passage_attention = SymmetricSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
                                                                   scale_dot_products = True, activation = torch.nn.ReLU,
                                                                   linear_layer_weight_init = kaiming_normal_)

        self.bidirectional_attention_module_2 = BidirectionalAttention(self.final_embedding_size * 2,
                                                                       return_with_inputs_concatenated = False, activation = torch.nn.Tanh,
                                                                       scale_dot_products = True)

        # self.bidirectional_attention_module_2 = BidafAttention(self.final_embedding_size * 2,
        #                                                                return_with_inputs_concatenated = False, activation = torch.nn.Tanh)

        self.linear_layer_before_final_passage_rep = torch.nn.Linear(self.final_embedding_size * 4, self.final_embedding_size * 2)
        self.norm_layer_before_final_passage_rep = torch.nn.BatchNorm1d(self.final_embedding_size * 2)

        xavier_normal_(self.linear_layer_before_final_passage_rep.weight)

        self.lstm_for_final_passage_representation_start_index = torch.nn.LSTM(self.final_embedding_size * 2, self.final_embedding_size * 2,
                                                                               bidirectional = True, batch_first = True, bias = True)

        self.lstm_for_final_passage_representation_end_index = torch.nn.LSTM(self.final_embedding_size * 2 + 1, self.final_embedding_size * 2,
                                                                               bidirectional = True, batch_first = True, bias = True)

        init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_start_index, value = 1.0)
        init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation_end_index, value = 1.0)

        self.lstm_for_final_passage_rep_init_hidden_state, self.lstm_for_final_passage_rep_init_cell_state = \
            self._init_lstm_hidden_and_cell_state(1, self.final_embedding_size * 2, is_bidirectional = True, as_trainable_params = True)

        self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
        self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)

        self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
        self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)

    def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
        question_batch = self.embedding(question_batch_index_tensor)
        passage_batch = self.embedding(passage_batch_index_tensor)

        # pdb.set_trace()

        # using learned initial states for the embedding lstm
        # embedding_init_states = (self.lstm_for_embedding_init_hidden_state.repeat(1, question_batch.size()[0], 1),
        #  self.lstm_for_embedding_init_cell_state.repeat(1, question_batch.size()[0], 1))
        #
        # # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
        # question_lstm_output = self.lstm_for_embedding(question_batch, embedding_init_states)[0]
        #
        # passage_lstm_output = self.lstm_for_embedding(passage_batch, embedding_init_states)[0]

        question_lstm_output = self.lstm_for_embedding(question_batch, self._init_lstm_hidden_and_cell_state(
            question_batch.size()[0], self.embedding_lstm_hidden_size))[0]

        passage_lstm_output = self.lstm_for_embedding(passage_batch, self._init_lstm_hidden_and_cell_state(
            passage_batch.size()[0], self.embedding_lstm_hidden_size))[0]

        passage_embedding_input = torch.cat([passage_batch, passage_lstm_output], dim = 2)
        question_embedding_input = torch.cat([question_batch, question_lstm_output], dim = 2)

        # (N, seq_length, embedding_size)
        passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
            torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
        question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
            self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)

        # (N, seq_length,  2 * embedding_size)
        question_bidirectional_attention_output, passage_bidirectional_attention_output = \
            self.bidirectional_attention_module(question_embedding, passage_embedding)

        # each is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
        # input with the bi-att output
        question_self_bi_att_output = self.question_to_question_attention(question_embedding)
        passage_self_bi_att_output = self.passage_to_passage_attention(passage_embedding)

        # (N, length, 2 * INPUT_SIZE) each
        question_bidirectional_attention_output_2, passage_bidirectional_attention_output_2 = \
            self.bidirectional_attention_module_2(question_self_bi_att_output, passage_self_bi_att_output)

        # (N, SEQUENCE_LENGTH, 2 * input_size)
        final_concatenated_passage_output = torch.transpose(self.linear_layer_before_final_passage_rep(torch.cat([passage_bidirectional_attention_output,
                                                                   passage_bidirectional_attention_output_2], dim = 2)),
                                                            dim0 = 1, dim1 = 2)

        final_concatenated_passage_output = torch.transpose(self.norm_layer_before_final_passage_rep(final_concatenated_passage_output), dim0 = 1, dim1 = 2)

        init_hidden_state_repeated = self.lstm_for_final_passage_rep_init_hidden_state.repeat(1, final_concatenated_passage_output.size()[0], 1)
        init_cell_state_repeated = self.lstm_for_final_passage_rep_init_cell_state.repeat(1, final_concatenated_passage_output.size()[0], 1)

        # start index
        final_enriched_passage_output_start_index = torch.cat(
            [passage_embedding, self.lstm_for_final_passage_representation_start_index(final_concatenated_passage_output, (init_hidden_state_repeated,
                                                                                                                           init_cell_state_repeated))[0]], dim = 2)

        final_enriched_passage_output_start_index = torch.transpose(final_enriched_passage_output_start_index, dim0 = 1, dim1 = 2)
        batch_normed_start_index_output = self.start_index_batch_norm_layer(final_enriched_passage_output_start_index)
        final_outputs_for_start_index = self.linear_layer_for_start_index(torch.transpose(batch_normed_start_index_output, dim0 = 1, dim1 = 2))


        # end index
        final_enriched_passage_output_end_index = torch.cat(
            [passage_embedding, self.lstm_for_final_passage_representation_end_index(torch.cat([final_concatenated_passage_output,
                                                                                                final_outputs_for_start_index], dim = 2),
                                                                                     (init_hidden_state_repeated, init_cell_state_repeated))[0]], dim = 2)

        final_enriched_passage_output_end_index = torch.transpose(final_enriched_passage_output_end_index, dim0 = 1, dim1 = 2)
        batch_normed_end_index_output = self.end_index_batch_norm_layer(final_enriched_passage_output_end_index)
        final_outputs_for_end_index = self.linear_layer_for_end_index(torch.transpose(batch_normed_end_index_output, dim0 = 1, dim1 = 2))

        return final_outputs_for_start_index, final_outputs_for_end_index


# class QAModuleWithAttentionLargeWithDiffBiAttnForStartVsEndIndex(Module):
#     """
#         same as QAModuleWithAttentionLarge but uses a different bi-attn #2 for start vs end-index. accuracy went down by 1% from 37.x -> 36.x. It could be the result of
#         2 epochs of training unable to train a wider model better
#     """
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__()
#         self.embedding = embedding
#         self.lstm_for_embedding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
#                                                 bidirectional = True, batch_first = True, bias = False)
#         # init_lstm_forget_gate_biases(self.lstm_for_embedding, value = 1.0)
#
#         self.final_embedding_size = int(self.embedding.embedding_dim)
#         self.embedding_activation = torch.tanh
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3, self.final_embedding_size)
#         xavier_normal_(self.linear_layer_for_final_embedding.weight)
#
#         self.norm_layer_for_embedding_output = torch.nn.BatchNorm1d(self.final_embedding_size)
#
#         self.bidirectional_attention_module = BidirectionalAttention(self.final_embedding_size, activation = torch.nn.Tanh, scale_dot_products = True)
#
#         self.question_to_question_attention = SymmetricSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                      scale_dot_products = True, activation = torch.nn.ReLU,
#                                                                      linear_layer_weight_init = kaiming_normal_)
#         self.passage_to_passage_attention = SymmetricSelfAttention(self.final_embedding_size, return_with_inputs_concatenated = True,
#                                                                    scale_dot_products = True, activation = torch.nn.ReLU,
#                                                                    linear_layer_weight_init = kaiming_normal_)
#
#         self.bidirectional_attention_module_2_start_index = BidirectionalAttention(self.final_embedding_size * 2,
#                                                                                    return_with_inputs_concatenated = False,
#                                                                                    activation = torch.nn.Tanh,
#                                                                                    scale_dot_products = True)
#
#         self.bidirectional_attention_module_2_end_index = BidirectionalAttention(self.final_embedding_size * 2,
#                                                                                  return_with_inputs_concatenated = False,
#                                                                                  activation = torch.nn.Tanh,
#                                                                                  scale_dot_products = True)
#
#         self.lstm_for_final_passage_representation = torch.nn.LSTM(self.final_embedding_size * 4, self.final_embedding_size * 2,
#                                                                    bidirectional = True, batch_first = True, bias = True)
#
#         init_lstm_forget_gate_biases(self.lstm_for_final_passage_representation, value = 1.0)
#
#         self.start_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#         self.end_index_batch_norm_layer = torch.nn.BatchNorm1d(self.final_embedding_size * 5)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.final_embedding_size * 5, 1)
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
#         passage_embedding = torch.transpose(self.norm_layer_for_embedding_output(
#             torch.transpose(self.linear_layer_for_final_embedding(passage_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#         question_embedding = torch.transpose(self.norm_layer_for_embedding_output(torch.transpose(
#             self.linear_layer_for_final_embedding(question_embedding_input), dim0 = 1, dim1 = 2)), dim0 = 1, dim1 = 2)
#
#         # (N, seq_length, 2 * embedding_size)
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module(question_embedding, passage_embedding)
#
#         # each is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
#         # input with the bi-att output
#         question_self_bi_att_output = self.question_to_question_attention(question_embedding)
#         passage_self_bi_att_output = self.passage_to_passage_attention(passage_embedding)
#
#         # (N, length, 2 * INPUT_SIZE) each
#         question_bidirectional_attention_output_2_start_index, passage_bidirectional_attention_output_2_start_index = \
#             self.bidirectional_attention_module_2_start_index(question_self_bi_att_output, passage_self_bi_att_output)
#
#         question_bidirectional_attention_output_2_end_index, passage_bidirectional_attention_output_2_end_index = \
#             self.bidirectional_attention_module_2_end_index(question_self_bi_att_output, passage_self_bi_att_output)
#
#         # (N, SEQUENCE_LENGTH, 4 * input_size)
#         final_concatenated_passage_output_start_index = torch.cat([passage_bidirectional_attention_output,
#                                                                    passage_bidirectional_attention_output_2_start_index], dim = 2)
#
#         final_concatenated_passage_output_end_index = torch.cat([passage_bidirectional_attention_output,
#                                                                  passage_bidirectional_attention_output_2_end_index], dim = 2)
#
#         final_enriched_passage_output_start_index = torch.cat(
#             [passage_embedding, self.lstm_for_final_passage_representation(final_concatenated_passage_output_start_index,
#                                                                            self.__init_lstm_hidden_and_cell_state(
#                                                                                final_concatenated_passage_output_start_index.size()[0],
#                                                                                self.final_embedding_size * 2, is_bidirectional = True))[0]], dim = 2)
#
#         final_enriched_passage_output_end_index = torch.cat(
#             [passage_embedding, self.lstm_for_final_passage_representation(final_concatenated_passage_output_end_index,
#                                                                            self.__init_lstm_hidden_and_cell_state(
#                                                                                final_concatenated_passage_output_end_index.size()[
#                                                                                    0], self.final_embedding_size * 2,
#                                                                                is_bidirectional = True))[0]], dim = 2)
#
#         batch_normed_start_index_output = self.start_index_batch_norm_layer(torch.transpose(final_enriched_passage_output_start_index, dim0 = 1, dim1 = 2))
#         batch_normed_end_index_output = self.end_index_batch_norm_layer(torch.transpose(final_enriched_passage_output_end_index, dim0 = 1, dim1 = 2))
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

# class QAModuleWithAttentionAndPosTagInputNoBert(Module):
#     """
#         adding pos tags
#     """
#     def __init__(self, embedding: torch.nn.Embedding, device, pos_tag_embedding_size: int):
#         super().__init__()
#         self.embedding = embedding
#
#         self.lstm_for_question_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
#                                                         bidirectional = True, batch_first = True, bias = False)
#         self.lstm_for_passage_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
#                                                        bidirectional = True, batch_first = True, bias = False)
#
#         self.linear_layer_for_final_embedding = torch.nn.Linear(self.embedding.embedding_dim * 3 + pos_tag_embedding_size, self.embedding.embedding_dim)
#
#         self.bidirectional_attention_module = BidirectionalAttention(self.embedding.embedding_dim)
#         self.passage_to_passage_attention = BidirectionalAttention(self.embedding.embedding_dim)
#         self.question_to_question_attention = BidirectionalAttention(self.embedding.embedding_dim)
#
#         self.bidirectional_attention_module_2 = BidirectionalAttention(self.embedding.embedding_dim * 2,
#                                                                        return_with_inputs_concatenated = False)
#
#         self.lstm_for_final_passage_representation = torch.nn.LSTM(self.embedding.embedding_dim * 4, self.embedding.embedding_dim * 2,
#                                                                    bidirectional = True, batch_first = True, bias = False)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1, bias = False)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 5, 1, bias = False)
#
#         self.device = device
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor, question_pos_tags_tensor, passage_pos_tags_tensor):
#
#         question_batch = self.embedding(question_batch_index_tensor)
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # the pytorch lstm outputs: output, (h_n, c_n). the output size for these lstms is 2 * input-size (due to being bidirectional)
#         question_lstm_output = self.lstm_for_question_encoding(question_batch, self.__init_lstm_hidden_and_cell_state(
#             question_batch.size()[0], self.embedding.embedding_dim))[0]
#
#         passage_lstm_output = self.lstm_for_passage_encoding(passage_batch, self.__init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding.embedding_dim))[0]
#
#         # (N, seq_length, embedding_size)
#         passage_embedding = self.linear_layer_for_final_embedding(torch.cat([passage_batch, passage_pos_tags_tensor, passage_lstm_output], dim = 2))
#         question_embedding = self.linear_layer_for_final_embedding(torch.cat([question_batch, question_pos_tags_tensor, question_lstm_output], dim = 2))
#
#         # (N, seq_length, 2 * embedding_size)
#         question_bidirectional_attention_output, passage_bidirectional_attention_output = \
#             self.bidirectional_attention_module(question_embedding, passage_embedding)
#
#         # each part of the tuples is (N, length, 2 * INPUT_SIZE). 2 * because by-default the bi-att module concatenates
#         # input with the bi-att output
#         passage_self_bi_att_output_lhs, passage_self_bi_att_output_rhs = self.passage_to_passage_attention(passage_embedding, passage_embedding)
#         question_self_bi_att_output_lhs, question_self_bi_att_output_rhs = self.question_to_question_attention(question_embedding, question_embedding)
#
#         # (N, length, 2 * INPUT_SIZE)
#         passage_self_bi_att_output = (passage_self_bi_att_output_lhs + passage_self_bi_att_output_rhs)/2.0
#         question_self_bi_att_output = (question_self_bi_att_output_lhs + question_self_bi_att_output_rhs)/2.0
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
#                                                                                    self.__init_lstm_hidden_and_cell_state(
#                                                                  final_concatenated_passage_output.size()[0],
#                                                                  self.embedding.embedding_dim * 2, is_bidirectional = True))[0]], dim = 2)
#
#         final_outputs_for_start_index = self.linear_layer_for_start_index(final_enriched_passage_output)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(final_enriched_passage_output)
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
#
#
# class QAModuleWithAttentionNoBertPassageOnly(Module):
#     """
#         this is kinda insane but is meant to detect bugs. The expectation is that this model performs
#         very poorly
#     """
#
#     def __init__(self, embedding: torch.nn.Embedding, device):
#         super().__init__()
#         self.embedding = embedding
#         self.lstm_for_passage_encoding = torch.nn.LSTM(self.embedding.embedding_dim, self.embedding.embedding_dim,
#                                                        bidirectional = True, batch_first = True, bias = False)
#
#         self.linear_layer_for_start_index = torch.nn.Linear(self.embedding.embedding_dim * 2, 1, bias = False)
#         self.linear_layer_for_end_index = torch.nn.Linear(self.embedding.embedding_dim * 2, 1, bias = False)
#
#         self.device = device
#
#     def forward(self, question_batch_index_tensor: torch.Tensor, passage_batch_index_tensor: torch.Tensor):
#
#         passage_batch = self.embedding(passage_batch_index_tensor)
#
#         # the pytorch lstm outputs: output, (h_n, c_n)
#         passage_lstm_output = self.lstm_for_passage_encoding(passage_batch, self.__init_lstm_hidden_and_cell_state(
#             passage_batch.size()[0], self.embedding.embedding_dim))[0]
#
#         final_outputs_for_start_index = self.linear_layer_for_start_index(passage_lstm_output)
#         final_outputs_for_end_index = self.linear_layer_for_end_index(passage_lstm_output)
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
