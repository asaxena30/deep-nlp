import math
from collections import namedtuple
from typing import Dict
from typing import List, NamedTuple

import spacy
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time
import pickle

from src.data.dataset.dataset import SquadDataset
from src.data.dataset.datasetreaders import SquadReader
from src.data.instance.instance import QAInstanceWithAnswerSpan
from src.modules.question_answering_modules import QAModuleWithAttentionNoBert
from src.tokenization.tokenizers import SpacyTokenizer
from src.util import datasetutils

dataset_data_file_path: str = "../../data/SQuAD"
serialized_dataset_file_path: str = "../../data/squad_serialized"

# can be turned to true once you have the training dataset pickled. Setting this to true and making sure
# the serialized_dataset_file_path, serialized_training_data_file_path and serialized_dev_data_file_path are
# accurate should make sure the training/dev datasets are loaded from the correct location
use_serialized_datasets: bool = False

# training_data_file_path: str = dataset_data_file_path + "/train-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

# training_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

training_data_file_path: str = dataset_data_file_path + "/sample.json"
dev_data_file_path: str = dataset_data_file_path + "/sample.json"

serialized_training_data_file_path: str = serialized_dataset_file_path + "/squad_dataset_train"
serialized_dev_data_file_path: str = serialized_dataset_file_path + "/squad_dataset_dev"




# the number of epochs to train the model for
NUM_EPOCHS: int = 2

# batch size to use for training. The default can be kept small when using gradient accumulation. However, if
# use_gradient_accumulation were to be False, it's best to use a reasonably large batch size
BATCH_SIZE: int = 50

WORD_EMBEDDING_SIZE = 300

fasttext_file_path: str = "/Users/asaxena/Downloads/fasttext_300d/fasttext-300d.vec"
fasttext_vectors_as_ordered_dict: Dict[str, torch.Tensor] = datasetutils.load_word_vectors_as_ordered_dict(fasttext_file_path, expected_embedding_size = WORD_EMBEDDING_SIZE)

answer_start_marker = "π"
answer_end_marker = "ß"

answer_start_marker_with_spaces: str = " %s " % answer_start_marker
answer_end_marker_with_spaces: str = " %s " % answer_end_marker

spacy_nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = SpacyTokenizer(spacy_nlp)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_serialized_dataset: bool = True

SquadTuple: NamedTuple = namedtuple('SquadTuple', ['question_tokens', 'passage_tokens', 'answer_start_index',
                                                   'answer_end_index', 'answer'])


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

        # print(passage_tokens)
        # print(question_tokens)
        # print(squad_qa_instance_as_dict['answer'])

    return SquadDataset(sorted(squad_dataset_list, key = lambda x: x.total_length))


def load_serialized_dataset(datafile_path: str) -> SquadDataset:
    with open(datafile_path, "rb") as f:
        return pickle.load(f)


embedding_weights = torch.cat([val.unsqueeze(dim = 0) for val in fasttext_vectors_as_ordered_dict.values()] +
                              [torch.zeros(1, 300)])

num_embeddings = embedding_weights.shape[0]

embedding = torch.nn.Embedding(embedding_dim = WORD_EMBEDDING_SIZE, num_embeddings = num_embeddings,
                               _weight = embedding_weights)

embedding_index_for_unknown_words = torch.tensor([num_embeddings - 1], dtype = torch.long)

words_to_index_dict = {key: index for index, key in enumerate(fasttext_vectors_as_ordered_dict.keys())}

print("loading training dataset...., time = " + str(time.time()))
train_dataset = load_serialized_dataset(serialized_training_data_file_path) if use_serialized_datasets else\
    get_squad_dataset_from_file(training_data_file_path)
print("train dataset loaded, time = " + str(time.time()))

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding, shuffle = True)

qa_module = QAModuleWithAttentionNoBert(embedding, token_to_index_dict = words_to_index_dict,
                                        embedding_index_for_unknown_words = num_embeddings - 1, device = device)

qa_module.to(device = device)
loss_function = torch.nn.CrossEntropyLoss()

num_train_iterations = NUM_EPOCHS * math.ceil(len(train_dataset) / BATCH_SIZE)

optimizer = torch.optim.Adam(qa_module.parameters(), lr = 5e-3)

# cyclic LR doesn't work with Adam for pytorch 1.1 due to a pytorch bug. also, cyclic LR may not play well with adaptive leanring rates
# scheduler = CyclicLR(optimizer, base_lr = 3e-5, max_lr = 0.01, step_size_up = 10, step_size_down = 90, cycle_momentum = False)

iteration_count: int = 0

for epoch in tqdm(range(NUM_EPOCHS)):
    print("inside training loop")
    for batch in train_dataloader:
        qa_module.zero_grad()
        start_index_outputs, end_index_outputs = qa_module(batch)

        answer_start_and_end_indices_original = torch.tensor([instance.answer_start_and_end_index for instance in batch], dtype = torch.long)
        answer_start_indices_original, answer_end_indices_original = torch.chunk(answer_start_and_end_indices_original, chunks = 2, dim = 1)

        start_index_loss = loss_function(start_index_outputs, answer_start_indices_original.to(device = device))
        end_index_loss = loss_function(end_index_outputs, answer_end_indices_original.to(device = device))

        total_loss = start_index_loss + end_index_loss

        if iteration_count % 10 == 0:
            print(total_loss)

        total_loss.backward()
        # scheduler.step()
        optimizer.step()
        iteration_count += 1


qa_module.eval()

torch.cuda.empty_cache()

test_dataset = load_serialized_dataset(serialized_dev_data_file_path) if use_serialized_datasets else\
    get_squad_dataset_from_file(dev_data_file_path)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding)

num_correct_start_index_answers: int = 0
num_correct_end_index_answers: int = 0
total_answers: int = 0

with torch.no_grad():
    for batch in test_dataloader:

        start_index_outputs, end_index_outputs = qa_module(batch)

        answer_start_and_end_indices_original = torch.tensor(
            [instance.answer_start_and_end_index for instance in batch], dtype = torch.long)
        answer_start_indices_original, answer_end_indices_original = torch.chunk(answer_start_and_end_indices_original.to(device = device),
                                                                                 chunks = 2, dim = 1)
        start_index_loss = loss_function(start_index_outputs, answer_start_indices_original)
        end_index_loss = loss_function(end_index_outputs, answer_end_indices_original)

        total_loss = start_index_loss + end_index_loss

        if iteration_count % 10 == 0:
            print("test loss at iteration# " + str(iteration_count) + " = " + str(total_loss))
            iteration_count += 1

        answer_start_indices_chosen_by_model = torch.squeeze(torch.topk(start_index_outputs, 1, dim = 1)[1])
        answer_end_indices_chosen_by_model = torch.squeeze(torch.topk(end_index_outputs, 1, dim = 1)[1])

        # torch.squeeze is used to make sure original indices matrices are squeezed to dimension (N)
        # as opposed to (N, 1) as answer_indices_chosen_by_model are vectors of size (N) due to which
        # a comparison with a matrix of size (N, 1) confuses torch.topk making ir return incorrect results
        answer_start_index_comparison_tensor = torch.eq(torch.squeeze(answer_start_indices_original),
                                                        answer_start_indices_chosen_by_model)
        answer_end_index_comparison_tensor = torch.eq(torch.squeeze(answer_end_indices_original),
                                                      answer_end_indices_chosen_by_model)

        num_correct_start_index_answers += answer_start_index_comparison_tensor.sum().item()
        num_correct_end_index_answers += answer_end_index_comparison_tensor.sum().item()
        total_answers += len(batch)

print("start index accuracy: " + str(num_correct_start_index_answers / total_answers))
print("end index accuracy: " + str(num_correct_end_index_answers / total_answers))
print("total answers: " + str(total_answers))

torch.save(qa_module.state_dict(), "./qa_module")
