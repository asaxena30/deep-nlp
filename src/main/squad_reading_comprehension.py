import math
import collections
from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time
import pickle
import pdb

from src.data.dataset.dataset import SquadDataset
from src.data.dataset.datasetreaders import SquadReader
from src.data.instance.instance import QAInstanceWithAnswerSpan
from src.modules.question_answering_modules import QAModuleWithAttentionLarge
from src.optim.adamw import AdamW
from src.optim.cycliclr import CyclicLR
from src.optim.radam import RAdam
from src.util.vector_encoding_utils import build_index_tensor_for_tokenized_sentences
from src.util import vector_encoding_utils
from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors


dataset_data_file_path: str = "../../data/SQuAD"
serialized_dataset_file_path: str = "../../data/squad_serialized"
FASTTEXT = "fasttext"
GLOVE = "glove"

# can be turned to true once you have the training dataset pickled. Setting this to true and making sure
# the serialized_dataset_file_path, serialized_training_data_file_path and serialized_dev_data_file_path are
# accurate should make sure the training/dev datasets are loaded from the correct location
use_serialized_datasets: bool = False
use_serialized_model: bool = False
skip_model_training: bool = True

embedding_type: str = GLOVE

# training_data_file_path: str = dataset_data_file_path + "/train-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

training_data_file_path: str = dataset_data_file_path + "/sample.json"
dev_data_file_path: str = dataset_data_file_path + "/sample.json"


serialized_training_data_file_path: str = serialized_dataset_file_path + "/squad_dataset_train"
serialized_dev_data_file_path: str = serialized_dataset_file_path + "/squad_dataset_dev"

serialized_model_file_path = "../../saved_models/qa_module"


# the number of epochs to train the model for
NUM_EPOCHS: int = 2

# batch size to use for training. The default can be kept small when using gradient accumulation. However, if
# use_gradient_accumulation were to be False, it's best to use a reasonably large batch size
BATCH_SIZE: int = 2

WORD_EMBEDDING_SIZE = 300

# fasttext_file_path: str = "/Users/asaxena/Downloads/fasttext_300d/fasttext-300d.vec"
# fasttext_vectors_as_ordered_dict: Dict[str, torch.Tensor] = datasetutils.load_word_vectors_as_ordered_dict(fasttext_file_path, expected_embedding_size = WORD_EMBEDDING_SIZE)
fasttext_serialized_file_path = "../../data/pickled/embedding_dicts/fasttext_serialized/fasttext_word_vectors_enhanced"

one_dim_zero_vector = torch.zeros(300)
zero_word_vector = one_dim_zero_vector.unsqueeze(dim = 0)

padding_token: str = '<pad>'

word_vectors_as_ordered_dict: Dict[str, torch.Tensor] = None

if embedding_type == FASTTEXT:
    with open(fasttext_serialized_file_path, "rb") as f:
        word_vectors_as_ordered_dict = pickle.load(f)
elif embedding_type == GLOVE:
    print("loading glove")
    glove = _PretrainedWordVectors(cache = "../../glove_vectors_trimmed/", name = "trimmed_vectors.txt")
    word_vectors_as_ordered_dict = {token: glove.vectors[idx] for token, idx in glove.token_to_index.items()}

word_vectors_as_ordered_dict[padding_token] = one_dim_zero_vector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

universal_pos_tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', '.', 'X', 'UNK']
null_tag_tensor = torch.zeros((1, 1, len(universal_pos_tagset)))

pos_tags_to_tensor_dict: Dict[str, torch.Tensor] = vector_encoding_utils.build_one_hot_encoded_tensors_for_tags(universal_pos_tagset)


def collate_with_padding(batch):
    if not isinstance(batch[0], QAInstanceWithAnswerSpan):
        raise NotImplementedError("only a QAInstanceWithAnswerSpan/subclass is supported")

    max_question_length_instance = max(batch, key = lambda batch_item: len(batch_item.question))
    max_question_length = len(max_question_length_instance.question)

    max_passage_length_instance = max(batch, key = lambda batch_item: len(batch_item.passage))
    max_passage_length = len(max_passage_length_instance.passage)
    new_batch = []

    for instance in batch:
        instance_question_length = len(instance.question)
        question_padding_size = max_question_length - instance_question_length
        instance.question = instance.question + [padding_token] * question_padding_size

        instance_passage_length = len(instance.passage)
        passage_padding_size = max_passage_length - instance_passage_length
        instance.passage = instance.passage + [padding_token] * passage_padding_size

        # batch_as_tuples.append(
        #     SquadTuple(instance.question, instance.passage, instance.answer_start_and_end_index[0],
        #                instance.answer_start_and_end_index[1],
        #                instance.answer))
        new_batch.append(instance)
    return new_batch


def load_serialized_dataset(datafile_path: str) -> SquadDataset:
    with open(datafile_path, "rb") as f:
        return pickle.load(f)


def calculate_loss_scaling_factor(original_start_indices, original_end_indices,
                                  model_start_index_scores, model_end_index_scores):
    start_indices_from_model = torch.max(model_start_index_scores, dim = 1)[1]
    end_indices_from_model = torch.max(model_end_index_scores, dim = 1)[1]

    lower_index_max = torch.max(original_start_indices, start_indices_from_model)
    upper_index_min = torch.min(original_end_indices, end_indices_from_model)

    # note: we need this code to work with pytorch 1.0 which is why the older version of
    # clamp is being used which forces specifying both a min and max. The max value here is artificial and specific
    # to this dataset
    overlap_range_length_tensor = torch.clamp(upper_index_min - lower_index_max + 1, min = 0, max = 10000)

    original_answer_length_tensor = original_end_indices - original_start_indices + 1
    model_answer_length_tensor = torch.abs(end_indices_from_model - start_indices_from_model) + 1

    # basically 1 - 2 * overlap_range_length/(actual_answer_length + model_answer_length)
    return 1 - torch.mean(torch.div(2 * overlap_range_length_tensor.float(), (original_answer_length_tensor + model_answer_length_tensor).float()))


def print_word_vector(passage_tokens):

    for token in passage_tokens:
        if token not in word_vectors_as_ordered_dict:
            print("word = " + token + " not found")


embedding_weights = torch.cat([val.unsqueeze(dim = 0) if type(val) == torch.Tensor else zero_word_vector for val in word_vectors_as_ordered_dict.values()] +
                              [zero_word_vector])

num_embeddings = embedding_weights.shape[0]

embedding = torch.nn.Embedding(embedding_dim = WORD_EMBEDDING_SIZE, num_embeddings = num_embeddings,
                               _weight = embedding_weights)

embedding_index_for_unknown_words = torch.tensor([num_embeddings - 1], dtype = torch.long)

words_to_index_dict = {key: index for index, key in enumerate(word_vectors_as_ordered_dict.keys())}
# pdb.set_trace()


print("loading training dataset...., time = " + str(time.time()))
train_dataset = load_serialized_dataset(serialized_training_data_file_path) if use_serialized_datasets else\
    SquadReader.get_squad_dataset_from_file(training_data_file_path)

print("train dataset loaded, time = " + str(time.time()))

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding, shuffle = True)

# the method of feeding inputs to the module from #1  - #3 makes models incompatible with onnx
# qa_module = QAModuleWithAttentionNoBert3(embedding, token_to_index_dict = words_to_index_dict,
#                                         embedding_index_for_unknown_words = num_embeddings - 1, device = device)


qa_module = QAModuleWithAttentionLarge(embedding, device = device)

if use_serialized_model:
    qa_module.load_state_dict(torch.load(serialized_model_file_path))

qa_module.to(device = device)
loss_function = torch.nn.CrossEntropyLoss()

num_train_iterations = NUM_EPOCHS * math.ceil(len(train_dataset) / BATCH_SIZE)
iteration_count: int = 0
training_iteration_to_loss_dict = collections.OrderedDict()
iteration_count_for_error_plot = num_train_iterations/10

optimizer = torch.optim.Adam(qa_module.parameters(), lr = 1e-3)

#optimizer = RAdam(qa_module.parameters(), lr = 1e-3)

# lr_scheduler = CyclicLR(optimizer, base_lr = 5e-5, max_lr = 1e-3, step_size_up = num_cycle_warmup_iterations,
#                         step_size_down = num_cycle_cooldown_iterations, cycle_momentum = False)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.1)
num_cycle_warmup_iterations: int = math.floor(num_train_iterations/5)
num_cycle_cooldown_iterations: int = num_train_iterations - num_cycle_warmup_iterations

if not skip_model_training:
    for epoch in tqdm(range(NUM_EPOCHS)):
        print("inside training loop")
        lr_scheduler.step()
        for batch in train_dataloader:

            qa_module.zero_grad()

            question_batch_index_tensor: torch.Tensor = build_index_tensor_for_tokenized_sentences(
                    tokenized_sentence_list = [instance.question for instance in batch],
                    token_to_index_dict = words_to_index_dict,
                    index_for_unknown_tokens = num_embeddings - 1).to(device = device)

            passage_batch_index_tensor: torch.Tensor = build_index_tensor_for_tokenized_sentences(
                    tokenized_sentence_list = [instance.passage for instance in batch],
                    token_to_index_dict = words_to_index_dict,
                    index_for_unknown_tokens = num_embeddings - 1).to(device = device)

            start_index_outputs, end_index_outputs = qa_module(question_batch_index_tensor, passage_batch_index_tensor)

            batch_start_and_end_indices = [instance.answer_start_and_end_index for instance in batch]
            answer_start_and_end_indices_original = torch.tensor(batch_start_and_end_indices, dtype = torch.long)

            answer_start_indices_original, answer_end_indices_original = torch.chunk(answer_start_and_end_indices_original, chunks = 2, dim = 1)

            start_index_loss = loss_function(start_index_outputs, answer_start_indices_original.to(device = device))
            end_index_loss = loss_function(end_index_outputs, answer_end_indices_original.to(device = device))

            total_loss = (start_index_loss + end_index_loss) * calculate_loss_scaling_factor(answer_start_indices_original, answer_end_indices_original, start_index_outputs, end_index_outputs)

            if iteration_count % 10 == 0:
                print(total_loss)

            if iteration_count % iteration_count_for_error_plot == 0:
                training_iteration_to_loss_dict[iteration_count] = total_loss.data.item()

            total_loss.backward()
            # scheduler.step()
            optimizer.step()
            iteration_count += 1


qa_module.eval()

# torch.cuda.empty_cache()

test_dataset = load_serialized_dataset(serialized_dev_data_file_path) if use_serialized_datasets else\
    SquadReader.get_squad_dataset_from_file(dev_data_file_path)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding)

num_correct_start_index_answers: int = 0
num_correct_end_index_answers: int = 0
num_answers_with_both_indices_correct: int = 0
total_answers: int = 0

test_iteration_to_loss_dict = collections.OrderedDict()

num_test_iterations = math.ceil(len(test_dataset) / BATCH_SIZE)
iteration_count_for_error_plot = num_test_iterations/10

# reset the iteration count
iteration_count = 0
predictions_for_eval: Dict = {}

with torch.no_grad():
    for batch in test_dataloader:

        for instance in batch:
            print_word_vector(instance.passage)

        question_batch_index_tensor: torch.Tensor = build_index_tensor_for_tokenized_sentences(
            tokenized_sentence_list = [instance.question for instance in batch],
            token_to_index_dict = words_to_index_dict,
            index_for_unknown_tokens = num_embeddings - 1).to(device = device)

        passage_batch_index_tensor: torch.Tensor = build_index_tensor_for_tokenized_sentences(
            tokenized_sentence_list = [instance.passage for instance in batch],
            token_to_index_dict = words_to_index_dict,
            index_for_unknown_tokens = num_embeddings - 1).to(device = device)

        start_index_outputs, end_index_outputs = qa_module(question_batch_index_tensor, passage_batch_index_tensor)

        answer_start_and_end_indices_original = torch.tensor(
            [instance.answer_start_and_end_index for instance in batch], dtype = torch.long)
        answer_start_indices_original, answer_end_indices_original = torch.chunk(answer_start_and_end_indices_original.to(device = device),
                                                                                 chunks = 2, dim = 1)
        start_index_loss = loss_function(start_index_outputs, answer_start_indices_original)
        end_index_loss = loss_function(end_index_outputs, answer_end_indices_original)

        total_loss = start_index_loss + end_index_loss

        original_answers_with_answers_inferred_from_indices = [(instance.answer,
                                                                instance.passage[instance.answer_start_and_end_index[0]:
                                                                                 instance.answer_start_and_end_index[1] + 1])
                                                               for instance in batch]

        print(original_answers_with_answers_inferred_from_indices)

        if iteration_count % 10 == 0:
            print("test loss at iteration# " + str(iteration_count) + " = " + str(total_loss))

        iteration_count += 1

        answer_start_indices_chosen_by_model = torch.squeeze(torch.topk(start_index_outputs, 1, dim = 1)[1])
        answer_end_indices_chosen_by_model = torch.squeeze(torch.topk(end_index_outputs, 1, dim = 1)[1])
        # pdb.set_trace()

        # predictions_for_eval.update({instance.id: instance.passage[answer_start_indices_chosen_by_model[i]:answer_end_indices_chosen_by_model[i] + 1]
        #                              for instance in batch for i in range(len(batch))})


        # torch.squeeze is used to make sure original indices matrices are squeezed to dimension (N)
        # as opposed to (N, 1) as answer_indices_chosen_by_model are vectors of size (N) due to which
        # a comparison with a matrix of size (N, 1) confuses torch.topk making it return incorrect results

        answer_start_index_comparison_tensor = torch.eq(torch.squeeze(answer_start_indices_original),
                                                        answer_start_indices_chosen_by_model)

        answer_end_index_comparison_tensor = torch.eq(torch.squeeze(answer_end_indices_original),
                                                      answer_end_indices_chosen_by_model)

        num_correct_start_index_answers += answer_start_index_comparison_tensor.sum().item()
        num_correct_end_index_answers += answer_end_index_comparison_tensor.sum().item()
        total_answers += len(batch)

        # print(answer_end_index_comparison_tensor)

        # print(answer_end_index_comparison_tensor)

        # now this should only count the values for which both answers are correct
        num_answers_with_both_indices_correct += (answer_start_index_comparison_tensor & answer_end_index_comparison_tensor).sum().item()

        if iteration_count % iteration_count_for_error_plot == 0:
            test_iteration_to_loss_dict[iteration_count] = total_loss.data.item()

        # answers_from_instance = [instance.answer for instance in batch]
        # answers_start_and_end_indices_chosen_by_model = zip(answer_start_indices_chosen_by_model.tolist(),
        #                                                     answer_end_indices_chosen_by_model.tolist())
        # answers_from_model_outputs = [" ".join(instance.passage[start_and_end_index[0]:start_and_end_index[1]] for start_and_end_index
        #                                          in answers_start_and_end_indices_chosen_by_model for instance in batch)]
        # print(answer_end_indices_chosen_by_model)
        # print([instance.answer_start_and_end_index for instance in batch])
        answer_start_indices_as_list = answer_start_indices_chosen_by_model.tolist()
        answer_end_indices_as_list = answer_end_indices_chosen_by_model.tolist()

        # this verified that the answers indeed correct
        # for idx in range(len(batch)):
        #     instance = batch[idx]
        #     print("answer = " + str(instance.answer))
        #     print(instance.passage[answer_start_indices_as_list[idx]:answer_end_indices_as_list[idx] + 1])


print("start index accuracy: " + str(num_correct_start_index_answers / total_answers))
print("end index accuracy: " + str(num_correct_end_index_answers / total_answers))
print("exact match accuracy: " + str(num_answers_with_both_indices_correct/total_answers))
print("total answers: " + str(total_answers))

# dev_dataset_as_json = SquadReader.extract_dataset(dev_data_file_path)


# test_loss_values_ordered = test_iteration_to_loss_dict.values()
# plt.plot(test_iteration_to_loss_dict.keys(), test_loss_values_ordered)
# plt.xlabel("test_step_num")
# plt.ylabel("test_loss")
# plt.show()

# visualization
# sample_input = test_dataset[0:2]
# sample_output = qa_module(sample_input)
# make_dot(sample_output, params = dict(qa_module.named_parameters()))
dummy_question_batch_input = torch.randint(low = 0, high = 12, size = (2, 12), device = device)
dummy_passage_batch_input = torch.randint(low = 0, high = 200, size = (2, 200), device = device)

#
# with SummaryWriter(comment='QAModule') as summaryWriter:
#     summaryWriter.add_graph(qa_module, (dummy_question_batch_input, dummy_passage_batch_input), True)

# torch.onnx.export(model = qa_module, args = (dummy_question_batch_input, dummy_passage_batch_input), f = "./qa_module3.onnx", verbose = True)
# torch.save(qa_module.state_dict(), "./qa_module")
