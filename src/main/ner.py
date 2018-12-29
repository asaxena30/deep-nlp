import torch
from src.util import datasetutils, vector_encoding_utils
from typing import Dict, Type
from src.sequence_models.lstmtagger import LSTMTagger
import torch.optim as optim
import collections
from random import shuffle
import matplotlib.pyplot as plt
from math import floor, fsum
from src.data.datasetreaders import Conll2003Reader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_data_file_path: str = "../../data/conll2003/eng.train"
test_data_file_path: str = "../../data/conll2003/eng.testa"

TensorType = Type[torch.Tensor]

word_index_in_line: int = 0
ner_tag_index_in_line: int = 3
word_vector_size: int = 300
hidden_state_size: int = 200
batch_size: int = 50

fasttext_file_path: str = "/Users/asaxena/Downloads/fasttext_300d/fasttext-300d.vec"
fasttext: Dict[str, torch.Tensor] = datasetutils.load_fasttext_vectors_as_dict(fasttext_file_path)

trim_ner_tag_prefixes: bool = False
ner_tags_with_prefixes: list = ["I-PER", "I-LOC", "B-LOC", "I-ORG", "B-ORG", "I-MISC", "B-MISC", "O"]
ner_tags_without_prefixes: list = ["PER", "LOC", "ORG", "MISC", "O"]

ner_tags: list = ner_tags_without_prefixes if trim_ner_tag_prefixes else ner_tags_with_prefixes

# using a multi-layered lstm didn't improve the mean test-loss much
ner_tagger = LSTMTagger(word_vector_size, hidden_state_size, True, 1, len(ner_tags))


def resolve_ner_tag_trim_if_necessary(ner_tag_with_prefix: str) -> str:
    return ner_tag_with_prefix[2:] if trim_ner_tag_prefixes and len(ner_tag_with_prefix) > 2 and \
                                      ner_tag_with_prefix[1] == '-' else ner_tag_with_prefix


training_sentences_with_tags = Conll2003Reader([training_data_file_path, test_data_file_path]).get_tagged_sentences()

all_sentence_batches_as_tensor_tuples = vector_encoding_utils.get_word_embeddings_for_tagged_sentences(training_sentences_with_tags, ner_tag_index_in_line, batch_size, ner_tags)

print("num batches: " + str(len(all_sentence_batches_as_tensor_tuples)))
for tagged_sentence_batch_with_labels in all_sentence_batches_as_tensor_tuples:
    print("shape for batch: " + str(tagged_sentence_batch_with_labels[0].shape) + "\n"
          + str(tagged_sentence_batch_with_labels[1].shape))


loss_function = torch.nn.BCELoss()
optimizer = optim.Adam(ner_tagger.parameters(), lr=0.01)

training_step_to_loss_dict = collections.OrderedDict()
training_step_num = 0

shuffle(all_sentence_batches_as_tensor_tuples)
training_sequence_length = floor(len(all_sentence_batches_as_tensor_tuples) * 0.8)
training_sentence_batches_as_tensor_tuples = all_sentence_batches_as_tensor_tuples[:training_sequence_length]

for epoch in range(0):
    shuffle(training_sentence_batches_as_tensor_tuples)
    for sentence_batch_tensor_with_labels in training_sentence_batches_as_tensor_tuples:
        ner_tagger.zero_grad()
        tag_scores = ner_tagger(sentence_batch_tensor_with_labels[0], sentence_batch_tensor_with_labels[0].shape[1]).permute(1, 0, 2)
        target_values = sentence_batch_tensor_with_labels[1].permute(1, 0, 2)
        loss = loss_function(tag_scores, target_values)
        loss.backward()
        # print(loss.data)
        training_step_to_loss_dict[training_step_num] = loss.data.item()
        optimizer.step()
        training_step_num += 1

# plt.plot(training_step_to_loss_dict.keys(), training_step_to_loss_dict.values())
# plt.xlabel("training_step_num")
# plt.ylabel("training_loss")
# plt.show()

test_sentence_batches_as_tensor_tuples = all_sentence_batches_as_tensor_tuples[training_sequence_length:]

mean_loss: int = 0
test_step_num: int = 0
test_step_to_loss_dict = collections.OrderedDict()
num_accurate_tags: int = 0
num_inaccurate_tags: int = 0

for sentence_batch_tensor_with_labels in test_sentence_batches_as_tensor_tuples:
    tag_scores = ner_tagger(sentence_batch_tensor_with_labels[0], sentence_batch_tensor_with_labels[0].shape[1]).permute(1, 0, 2)
    # print("tag score tensor:" + str(tag_scores.data))
    print("tag score tensor shape, test phase:" + str(tag_scores.data.shape))
    target_values = sentence_batch_tensor_with_labels[1].permute(1, 0, 2)

    index_for_one_as_row_tensor = target_values.nonzero()

    # label_from_target_value = target_values[index_for_one_as_row_tensor[0, 0].item(),
    #                                         index_for_one_as_row_tensor[0, 1].item(),
    #                                         index_for_one_as_row_tensor[0, 2].item()]
    # label_from_tag_scores = tag_scores[index_for_one_as_row_tensor[0, 0].item(),
    #                                    index_for_one_as_row_tensor[0, 1].item(),
    #                                    index_for_one_as_row_tensor[0, 2].item()]
    #
    # print("target_values shape, test phase:" + str(target_values.data.shape))
    #
    # print("comparing with target value: target = " + str(label_from_target_value) +
    #       ", tag_score for target_index = " + str(label_from_tag_scores))
    # # print("target_values tensor" + str(target_values.data))

    for dim_0_idx in range(tag_scores.shape[0]):
        for dim_1_idx in range(tag_scores.shape[1]):
            top_score_with_array_index = torch.topk(tag_scores[dim_0_idx, dim_1_idx], 1, dim=0)
            if target_values[dim_0_idx, dim_1_idx, int(top_score_with_array_index[1].item())].item() == 1:
                if num_accurate_tags <= 2:
                    print(target_values[dim_0_idx, dim_1_idx, int(top_score_with_array_index[1].item())].item())
                num_accurate_tags += 1
            else:
                num_inaccurate_tags += 1

    loss = loss_function(tag_scores, target_values)
    test_step_to_loss_dict[test_step_num] = loss.data.item()
    test_step_num += 1
    print(loss.data)

    # for sentence_num in range(sentence_batch_tensor_with_labels.shape[0]):
    #     expected_scores_for_sentence = target_values[sentence_num]
    #     actual_scores_for_sentence = tag_scores[sentence_num]


test_loss_values_ordered = test_step_to_loss_dict.values()
plt.plot(test_step_to_loss_dict.keys(), test_loss_values_ordered)
plt.xlabel("test_step_num")
plt.ylabel("test_loss")
plt.show()

print("mean test loss: " + str(fsum(test_loss_values_ordered) / len(test_loss_values_ordered)))
print("num accurate tags: " + str(num_accurate_tags))
print("num inaccurate tags: " + str(num_inaccurate_tags))

torch.save(ner_tagger.state_dict(keep_vars=True), "./ner_tagger.torch")


external_test_sentence_1 = "Inside the airport that bears his name, George Herbert Walker Bush looks, at a distance," \
                           " as if he’s wearing a cape".split()


external_test_sentence_2 = "The compromise, struck over a steak dinner at the Group of 20 meeting here and announced in" \
                           " a White House statement, was less a breakthrough than a breakdown averted".split()

embeddings_for_test_sentence_1 = vector_encoding_utils.get_word_embeddings_for_test_sentence(external_test_sentence_1)
embeddings_for_test_sentence_2 = vector_encoding_utils.get_word_embeddings_for_test_sentence(external_test_sentence_2)

print("embeddings_for_test_sentence_1:")
print(embeddings_for_test_sentence_1)
print("embeddings_for_test_sentence_2:")
print(embeddings_for_test_sentence_2)

sentence_embedding_tensor_1 = torch.cat(embeddings_for_test_sentence_1, 0).view(len(embeddings_for_test_sentence_1), 1, -1)
sentence_embedding_tensor_2 = torch.cat(embeddings_for_test_sentence_2, 0).view(len(embeddings_for_test_sentence_2), 1, -1)

tag_scores_test_1 = ner_tagger(sentence_embedding_tensor_1, sentence_embedding_tensor_1.shape[1])
tag_scores_test_2 = ner_tagger(sentence_embedding_tensor_2, sentence_embedding_tensor_2.shape[1])

print(tag_scores_test_1)
print(tag_scores_test_2)

words_with_tags_sentence_1 = []
words_with_tags_sentence_2 = []

for word_index in range(len(external_test_sentence_1)):
    word_tag = ner_tags[int(torch.topk(tag_scores_test_1[word_index, 0], 1, dim=0)[1].item())]
    words_with_tags_sentence_1.append((external_test_sentence_1[word_index], word_tag))

for word_index in range(len(external_test_sentence_2)):
    word_tag = ner_tags[int(torch.topk(tag_scores_test_2[word_index, 0], 1, dim=0)[1].item())]
    words_with_tags_sentence_2.append((external_test_sentence_2[word_index], word_tag))

print(words_with_tags_sentence_1)
print(words_with_tags_sentence_2)





