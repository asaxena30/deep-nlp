import math
from collections import namedtuple
from typing import Dict
from typing import List, NamedTuple

import spacy
import torch
import time
import pickle

from src.data.dataset.dataset import SquadDataset
from src.data.dataset.datasetreaders import SquadReader
from src.data.instance.instance import QAInstanceWithAnswerSpan
from src.tokenization.tokenizers import SpacyTokenizer
from src.util import datasetutils
from torch.utils.data.dataloader import DataLoader


dataset_data_file_path: str = "../../data/SQuAD"

# training_data_file_path: str = dataset_data_file_path + "/train-v2.0.json"
dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

# training_data_file_path: str = dataset_data_file_path + "/sample.json"
# dev_data_file_path: str = dataset_data_file_path + "/sample.json"

# training_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

answer_start_marker = "π"
answer_end_marker = "ß"

answer_start_marker_with_spaces: str = " %s " % answer_start_marker
answer_end_marker_with_spaces: str = " %s " % answer_end_marker

spacy_nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = SpacyTokenizer(spacy_nlp)

SquadTuple: NamedTuple = namedtuple('SquadTuple', ['question_tokens', 'passage_tokens', 'answer_start_index',
                                                   'answer_end_index', 'answer'])

BATCH_SIZE: int = 50

def collate_with_padding(batch):
    if not isinstance(batch[0], QAInstanceWithAnswerSpan):
        raise NotImplementedError("only a QAInstanceWithAnswerSpan is supported for this class")

    max_question_length_instance = max(batch, key = lambda batch_item: len(batch_item.question))
    max_question_length = len(max_question_length_instance.question)

    max_passage_length_instance = max(batch, key = lambda batch_item: len(batch_item.passage))
    max_passage_length = len(max_passage_length_instance.passage)
    batches = []

    for instance in batch:
        instance_question_length = len(instance.question)
        question_padding_size = max_question_length - instance_question_length
        instance.question = instance.question + ['pad'] * question_padding_size

        instance_passage_length = len(instance.passage)
        instance_padding_size = max_passage_length - instance_passage_length
        instance.passage = instance.passage + ['pad'] * instance_padding_size

        # batch_as_tuples.append(
        #     SquadTuple(instance.question, instance.passage, instance.answer_start_and_end_index[0],
        #                instance.answer_start_and_end_index[1],
        #                instance.answer))
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

        passage_tokens = spacy_tokenizer.tokenize(passage_text_for_tokenization)
        question_tokens = spacy_tokenizer.tokenize(squad_qa_instance_as_dict['question'])

        answer_start_marker_index = passage_tokens.index(answer_start_marker)
        answer_end_marker_index = passage_tokens.index(answer_end_marker)

        # let's remove the answer markers now
        passage_tokens = passage_tokens[:answer_start_marker_index] + \
                         passage_tokens[answer_start_marker_index + 1: answer_end_marker_index] + \
                         passage_tokens[answer_end_marker_index + 1:]

        answer_span_start_token_index: int = answer_start_marker_index

        # removing the start marker, shifts the answer towards the start by an additional index,
        # hence -2 as opposed to -1
        answer_span_end_token_index: int = answer_end_marker_index - 2

        answer_indices = (answer_span_start_token_index, answer_span_end_token_index)

        instance = QAInstanceWithAnswerSpan(question = question_tokens, passage = passage_tokens,
                                            answer = squad_qa_instance_as_dict['answer'],
                                            answer_start_and_end_index = answer_indices,
                                            total_length = len(question_tokens + passage_tokens))

        squad_dataset_list.append(instance)

    return SquadDataset(sorted(squad_dataset_list, key = lambda x: x.total_length))


print("loading dataset...., time = " + str(time.time()))
# train_dataset = get_squad_dataset_from_file(training_data_file_path)
dataset = get_squad_dataset_from_file(dev_data_file_path)
with open("./squad_dataset_dev", "wb+") as f:
    pickle.dump(dataset, f)

with open("./squad_dataset_dev", "rb") as f:
    dataset = pickle.load(f)

print("dataset loaded...., time = " + str(time.time()))

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding, shuffle = True)

print("dataloader built...., time = " + str(time.time()))

iteration_count: int = 0

for batch in dataloader:
    if iteration_count == 1:
        print(batch)
    iteration_count += 1
