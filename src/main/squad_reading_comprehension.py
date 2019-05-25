import math
from collections import namedtuple
from typing import Dict
from typing import List, NamedTuple

import spacy
import torch
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data.dataset.dataset import SquadDataset
from src.data.dataset.datasetreaders import SquadReader
from src.data.instance.instance import QAInstanceWithAnswerSpan
from src.modules.question_answering_modules import QAModuleWithAttentionNoBert
from src.tokenization.tokenizers import SpacyTokenizer
from src.util import datasetutils

dataset_data_file_path: str = "../../data/SQuAD"

# training_data_file_path: str = dataset_data_file_path + "/train-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

training_data_file_path: str = dataset_data_file_path + "/sample.json"
dev_data_file_path: str = dataset_data_file_path + "/sample.json"

# training_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"


# the number of epochs to train the model for
NUM_EPOCHS: int = 3

# batch size to use for training. The default can be kept small when using gradient accumulation. However, if
# use_gradient_accumulation were to be False, it's best to use a reasonably large batch size
BATCH_SIZE: int = 32

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


# note the additional zeros Tensor. This is done to assign an embedding of torch.zeros(300) to unknown words. Since the
# embedding is trainable, we expect this to change as training progresses
# embedding = torch.nn.Embedding(embedding_dim = 300, num_embeddings = num_embeddings,
#                                _weight = torch.cat([val.unsqueeze(dim = 0) for val in fasttext_vectors_as_ordered_dict.values()] +
#                                                                    [torch.zeros((1, 300))], 0))
# embedding_index_for_unknown_words = torch.tensor([num_embeddings - 1], dtype = torch.long)
#
# print(embedding[embedding_index_for_unknown_words])

embedding_weights = torch.cat([val.unsqueeze(dim = 0) for val in fasttext_vectors_as_ordered_dict.values()] +
                              [torch.zeros(1, 300)])

num_embeddings = embedding_weights.shape[0]

embedding = torch.nn.Embedding(embedding_dim = WORD_EMBEDDING_SIZE, num_embeddings = num_embeddings,
                               _weight = embedding_weights)

embedding_index_for_unknown_words = torch.tensor([num_embeddings - 1], dtype = torch.long)

# print(embedding(embedding_index_for_unknown_words))

words_to_index_dict = {key: index for index, key in enumerate(fasttext_vectors_as_ordered_dict.keys())}

train_dataset = get_squad_dataset_from_file(training_data_file_path)

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding, shuffle = True)

qa_module = QAModuleWithAttentionNoBert(embedding, token_to_index_dict = words_to_index_dict,
                                        embedding_index_for_unknown_words = num_embeddings - 1, device = device)

qa_module.to(device = device)
loss_function = torch.nn.CrossEntropyLoss()

num_train_iterations = NUM_EPOCHS * math.ceil(len(train_dataset) / BATCH_SIZE)

# optimizer = BertAdam(qa_module.parameters(), lr = 5e-5, warmup = 0.1, t_total = num_train_iterations)
optimizer = torch.optim.Adam(qa_module.parameters())
# scheduler = CyclicLR(optimizer, base_lr = 3e-5, max_lr = 0.01, step_size_up = 10, step_size_down = 90, cycle_momentum = False)

iteration_count: int = 0

for epoch in tqdm(range(NUM_EPOCHS)):
    print("inside training loop")
    for batch in train_dataloader:
        qa_module.zero_grad()
        start_index_outputs, end_index_outputs = qa_module(batch)

        answer_start_and_end_indices_original = torch.tensor([instance.answer_start_and_end_index for instance in batch], dtype = torch.long)
        answer_start_indices_original, answer_end_indices_original = torch.chunk(answer_start_and_end_indices_original, chunks = 2, dim = 1)

        start_index_loss = loss_function(start_index_outputs, torch.squeeze(answer_start_indices_original).to(device = device))
        end_index_loss = loss_function(end_index_outputs, torch.squeeze(answer_end_indices_original).to(device = device))
        total_loss = start_index_loss + end_index_loss

        if iteration_count % 10 == 0:
            print(total_loss)

        total_loss.backward()
        # scheduler.step()
        optimizer.step()
        iteration_count += 1


