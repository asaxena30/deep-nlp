from collections import namedtuple
from typing import List, NamedTuple, Dict

import time
import pickle

import nltk
from nltk.tag.perceptron import PerceptronTagger

from src.data.dataset.dataset import SquadDataset
from src.data.dataset.datasetreaders import SquadReader
from src.data.instance.instance import QAInstanceWithAnswerSpan, TaggedQAInstanceWithAnswerSpan
from nltk.tokenize import word_tokenize
import torch

from src.util import vector_encoding_utils

dataset_data_file_path: str = "../../data/SQuAD"

training_data_file_path: str = dataset_data_file_path + "/train-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

# training_data_file_path: str = dataset_data_file_path + "/sample.json"
# dev_data_file_path: str = dataset_data_file_path + "/sample.json"

# training_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

answer_start_marker = "π"
answer_end_marker = "ß"

answer_start_marker_with_spaces: str = " %s " % answer_start_marker
answer_end_marker_with_spaces: str = " %s " % answer_end_marker

SquadTuple: NamedTuple = namedtuple('SquadTuple', ['question_tokens', 'passage_tokens', 'answer_start_index',
                                                   'answer_end_index', 'answer'])

BATCH_SIZE: int = 10

universal_pos_tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', '.', 'X', 'UNK']
null_tag_tensor = torch.zeros((1, 1, len(universal_pos_tagset)))

pos_tags_to_tensor_dict: Dict[str, torch.Tensor] = vector_encoding_utils.build_one_hot_encoded_tensors_for_tags(universal_pos_tagset)

UNKNOWN_TAG = "UNK"

tagger = PerceptronTagger()


def collate_with_padding(batch):
    if not isinstance(batch[0], QAInstanceWithAnswerSpan):
        raise NotImplementedError("only a QAInstanceWithAnswerSpan/subclass is supported")

    max_question_length_instance = max(batch, key = lambda batch_item: len(batch_item.question))
    max_question_length = len(max_question_length_instance.question)

    max_passage_length_instance = max(batch, key = lambda batch_item: len(batch_item.passage))
    max_passage_length = len(max_passage_length_instance.passage)
    batches = []

    for instance in batch:
        instance_question_length = len(instance.question)
        question_padding_size = max_question_length - instance_question_length
        instance.question = instance.question + ['pad'] * question_padding_size

        if instance.question_pos_tags is not None:
            instance.question_pos_tags = torch.cat(instance.question_pos_tags + [null_tag_tensor] * question_padding_size, dim = 1)

        instance_passage_length = len(instance.passage)
        passage_padding_size = max_passage_length - instance_passage_length
        instance.passage = instance.passage + ['pad'] * passage_padding_size

        if instance.passage_pos_tags is not None:
            instance.passage_pos_tags = torch.cat(instance.passage_pos_tags + [null_tag_tensor] * passage_padding_size, dim = 1)

        batches.append(instance)
    return batches


def get_squad_dataset_from_file(file_path: str) -> SquadDataset:
    instances = SquadReader.read(file_path)
    squad_dataset_list: List = []

    for squad_qa_instance_as_dict in instances:
        if 'span_start' not in squad_qa_instance_as_dict:
            continue

        span_start_char_index: int = squad_qa_instance_as_dict['span_start']
        span_end_char_index: int = squad_qa_instance_as_dict['span_end']

        passage_text = squad_qa_instance_as_dict['passage']

        # here we are simply inserting answer start and end markers so that after tokenization we can still track
        # the answer boundaries
        passage_text_for_tokenization = passage_text[:span_start_char_index] + answer_start_marker_with_spaces + \
                                        passage_text[
                                        span_start_char_index: span_end_char_index] + answer_end_marker_with_spaces + \
                                        passage_text[span_end_char_index:]

        passage_tokens = word_tokenize(passage_text_for_tokenization)
        question_tokens = word_tokenize(squad_qa_instance_as_dict['question'])

        answer_start_marker_index = passage_tokens.index(answer_start_marker)
        answer_end_marker_index = passage_tokens.index(answer_end_marker)

        # let's remove the answer markers now
        passage_tokens = passage_tokens[:answer_start_marker_index] + \
                         passage_tokens[answer_start_marker_index + 1: answer_end_marker_index] + \
                         passage_tokens[answer_end_marker_index + 1:]

        pos_tags_for_passage_tokens = [pos_tags_to_tensor_dict[tagged_word[1]].view(1, 1, -1) for tagged_word in
                                       nltk.tag._pos_tag(passage_tokens, tagset = 'universal', tagger = tagger, lang = 'eng')]
        pos_tags_for_question_tokens = [pos_tags_to_tensor_dict[tagged_word[1]].view(1, 1, -1) for tagged_word in
                                        nltk.tag._pos_tag(question_tokens, tagset = 'universal', tagger = tagger, lang = 'eng')]

        answer_span_start_token_index: int = answer_start_marker_index

        # removing the start marker, shifts the answer towards the start by an additional index,
        # hence -2 as opposed to -1
        answer_span_end_token_index: int = answer_end_marker_index - 2

        answer_indices = (answer_span_start_token_index, answer_span_end_token_index)

        instance = TaggedQAInstanceWithAnswerSpan(question = question_tokens, question_pos_tags = pos_tags_for_question_tokens,
                                            passage = passage_tokens, passage_pos_tags = pos_tags_for_passage_tokens,
                                            answer = squad_qa_instance_as_dict['answer'],
                                            answer_start_and_end_index = answer_indices,
                                            total_length = len(question_tokens + passage_tokens))

        squad_dataset_list.append(instance)

        # print(passage_tokens)
        # print(question_tokens)
        # print(squad_qa_instance_as_dict['answer'])

    return SquadDataset(sorted(squad_dataset_list, key = lambda x: x.total_length))


print("loading dataset...., time = " + str(time.time()))
# train_dataset = get_squad_dataset_from_file(training_data_file_path)

dataset = get_squad_dataset_from_file(training_data_file_path)

iteration_count: int = 0

for instance in dataset:
    if iteration_count % 10000 == 0:
        print(instance.passage)
        print(instance.question)
    iteration_count += 1


print("serializing and re-loading dataset")

with open("./squad_dataset_serialized", "wb+") as f:
    pickle.dump(dataset, f)


with open("./squad_dataset_serialized", "rb") as f:
    serialized_dataset = pickle.load(f)

# print("dataset loaded...., time = " + str(time.time()))

iteration_count = 0

for instance in serialized_dataset:
    if iteration_count % 10000 == 0:
        print(instance.passage)
        print(instance.passage_pos_tags)
        print(instance.question)
        print(instance.question_pos_tags)
    iteration_count += 1

print(iteration_count)
