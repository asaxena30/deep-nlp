import collections
import math
import pickle
import time
from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors
from tqdm import tqdm

from src.common.neural_net_param_utils import add_parameter_stats_to_summary_writer
from src.data.dataset.datasetreaders import SquadReader
from src.main.squad_qa_helper import collate_with_padding, load_serialized_dataset, \
    get_qa_model_outputs, get_total_loss_and_original_indices
from src.modules.question_answering_modules import QAModuleOutputs, \
    QAModuleMod5StaticBatchNormsNoBNBeforeFinalLinearLayers, \
    QAModuleMod5StaticBatchNormsNoBNBeforeFinalLinearLayersWDropout
from src.util import vector_encoding_utils
from src.util.vector_encoding_utils import build_index_tensor_for_tokenized_sentences

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

serialized_model_file_path = "../../saved_models/qa_module_epoch_3_and_4_cyclic_lr_mod4"

# the number of epochs to train the model for
NUM_EPOCHS: int = 1

# batch size to use for training. The default can be kept small when using gradient accumulation. However, if
# use_gradient_accumulation were to be False, it's best to use a reasonably large batch size
BATCH_SIZE: int = 2  # 32

WORD_EMBEDDING_SIZE = 300

# fasttext_file_path: str = "/Users/asaxena/Downloads/fasttext_300d/fasttext-300d.vec"
# fasttext_vectors_as_ordered_dict: Dict[str, torch.Tensor] = datasetutils.load_word_vectors_as_ordered_dict(fasttext_file_path, expected_embedding_size = WORD_EMBEDDING_SIZE)
fasttext_serialized_file_path = "../../data/pickled/embedding_dicts/fasttext_serialized/fasttext_word_vectors_enhanced"

one_dim_zero_vector = torch.zeros(300)
zero_word_vector = one_dim_zero_vector.unsqueeze(dim=0)

padding_token: str = '<pad>'

word_vectors_as_ordered_dict: Dict[str, torch.Tensor] = None

if embedding_type == FASTTEXT:
    with open(fasttext_serialized_file_path, "rb") as f:
        word_vectors_as_ordered_dict = pickle.load(f)
elif embedding_type == GLOVE:
    print("loading glove")
    glove = _PretrainedWordVectors(cache="../../glove_vectors_trimmed_2/", name="trimmed_vectors_2.txt")
    word_vectors_as_ordered_dict = {token: glove.vectors[idx] for token, idx in glove.token_to_index.items()}

max_embedding_val = 0
for k, v in word_vectors_as_ordered_dict.items():
    max_embedding_val = max(v.data.max(), max_embedding_val)

print("max_embedding_val: " + str(max_embedding_val))

word_vectors_as_ordered_dict[padding_token] = one_dim_zero_vector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

universal_pos_tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', '.', 'X', 'UNK']
null_tag_tensor = torch.zeros((1, 1, len(universal_pos_tagset)))

pos_tags_to_tensor_dict: Dict[str, torch.Tensor] = vector_encoding_utils.build_one_hot_encoded_tensors_for_tags(
    universal_pos_tagset)

embedding_weights = torch.cat([val.unsqueeze(dim=0) if type(val) == torch.Tensor else zero_word_vector for val in
                               word_vectors_as_ordered_dict.values()] +
                              [zero_word_vector])

num_embeddings = embedding_weights.shape[0]

embedding = torch.nn.Embedding(embedding_dim=WORD_EMBEDDING_SIZE, num_embeddings=num_embeddings,
                               _weight=embedding_weights)

embedding_index_for_unknown_words = torch.tensor([num_embeddings - 1], dtype=torch.long)

words_to_index_dict = {key: index for index, key in enumerate(word_vectors_as_ordered_dict.keys())}
# pdb.set_trace()


print("loading training dataset...., time = " + str(time.time()))
train_dataset = load_serialized_dataset(serialized_training_data_file_path) if use_serialized_datasets else \
    SquadReader.get_squad_dataset_from_file(training_data_file_path, use_longest_answers=True)

print("train dataset loaded, time = " + str(time.time()))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_with_padding, shuffle=True)

summary_writer = SummaryWriter()

qa_module = QAModuleMod5StaticBatchNormsNoBNBeforeFinalLinearLayersWDropout(embedding, device=device,
                                                                    summary_writer=summary_writer)

if use_serialized_model:
    qa_module.load_state_dict(torch.load(serialized_model_file_path, map_location=device))

qa_module.to(device=device)
loss_function = torch.nn.CrossEntropyLoss()

num_train_iterations = NUM_EPOCHS * math.ceil(len(train_dataset) / BATCH_SIZE)
iteration_count: int = 0
training_iteration_to_loss_dict = collections.OrderedDict()
iteration_count_for_error_plot = num_train_iterations / 10

optimizer = torch.optim.Adam(qa_module.parameters(), lr=1e-3)

# num_cycle_warmup_iterations: int = math.ceil(num_train_iterations / 4)
# num_cycle_cooldown_iterations: int = num_train_iterations - num_cycle_warmup_iterations

# lr_scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=num_cycle_warmup_iterations,
#                         step_size_down=num_cycle_cooldown_iterations, cycle_momentum=False)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# num_cycle_warmup_iterations: int = math.floor(num_train_iterations / 5)
# num_cycle_cooldown_iterations: int = num_train_iterations - num_cycle_warmup_iterations

if not skip_model_training:
    for epoch in tqdm(range(NUM_EPOCHS)):
        print("inside training loop")
        for batch in train_dataloader:

            qa_module.zero_grad()

            qa_module_outputs: QAModuleOutputs = get_qa_model_outputs(batch, words_to_index_dict, num_embeddings,
                                                                      device, qa_module, iteration_count)

            start_index_outputs, end_index_outputs = qa_module_outputs.start_index_outputs, qa_module_outputs.end_index_outputs

            total_loss, answer_start_indices_original, answer_end_indices_original = \
                get_total_loss_and_original_indices(batch,
                                                    start_index_outputs,
                                                    end_index_outputs,
                                                    loss_function,
                                                    device)

            if qa_module_outputs.supplementary_loss:
                # print("total without supplementary loss: " + str(total_loss.data.item()))

                total_loss += qa_module_outputs.supplementary_loss

                # print("total with supplementary loss: " + str(total_loss.data.item()))

            if iteration_count % 10 == 0:
                print(total_loss)

            if iteration_count % iteration_count_for_error_plot == 0:
                training_iteration_to_loss_dict[iteration_count] = total_loss.data.item()

            add_parameter_stats_to_summary_writer(summary_writer, qa_module.named_parameters(), iteration_count)

            total_loss.backward()
            # scheduler.step()
            optimizer.step()
            iteration_count += 1
            lr_scheduler.step()

qa_module.eval()

# torch.cuda.empty_cache()

test_dataset = load_serialized_dataset(serialized_dev_data_file_path) if use_serialized_datasets else \
    SquadReader.get_squad_dataset_from_file(dev_data_file_path)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_with_padding)

num_correct_start_index_answers: int = 0
num_correct_end_index_answers: int = 0
num_answers_with_both_indices_correct: int = 0
total_answers: int = 0

test_iteration_to_loss_dict = collections.OrderedDict()

num_test_iterations = math.ceil(len(test_dataset) / BATCH_SIZE)
iteration_count_for_error_plot = num_test_iterations / 10

# reset the iteration count
iteration_count = 0
predictions_for_eval: Dict = {}

with torch.no_grad():
    for batch in test_dataloader:
        qa_module_outputs: QAModuleOutputs = get_qa_model_outputs(batch, words_to_index_dict, num_embeddings,
                                                                  device, qa_module, iteration_count)

        start_index_outputs, end_index_outputs = qa_module_outputs.start_index_outputs, qa_module_outputs.end_index_outputs

        total_loss, answer_start_indices_original, answer_end_indices_original = \
            get_total_loss_and_original_indices(
                batch,
                start_index_outputs,
                end_index_outputs,
                loss_function,
                device)

        original_answers_with_answers_inferred_from_indices = [(instance.answer,
                                                                instance.passage[instance.answer_start_and_end_index[0]:
                                                                                 instance.answer_start_and_end_index[
                                                                                     1] + 1])
                                                               for instance in batch]

        print(original_answers_with_answers_inferred_from_indices)

        if iteration_count % 10 == 0:
            print("test loss at iteration# " + str(iteration_count) + " = " + str(total_loss))

        iteration_count += 1

        answer_start_indices_chosen_by_model = torch.squeeze(torch.topk(start_index_outputs, 1, dim=1)[1])
        answer_end_indices_chosen_by_model = torch.squeeze(torch.topk(end_index_outputs, 1, dim=1)[1])
        # pdb.set_trace()

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
        num_answers_with_both_indices_correct += (
                answer_start_index_comparison_tensor & answer_end_index_comparison_tensor).sum().item()

        if iteration_count % iteration_count_for_error_plot == 0:
            test_iteration_to_loss_dict[iteration_count] = total_loss.data.item()

        answer_start_indices_as_list = answer_start_indices_chosen_by_model.tolist()
        answer_end_indices_as_list = answer_end_indices_chosen_by_model.tolist()

print("start index accuracy: " + str(num_correct_start_index_answers / total_answers))
print("end index accuracy: " + str(num_correct_end_index_answers / total_answers))
print("exact match accuracy: " + str(num_answers_with_both_indices_correct / total_answers))
print("total answers: " + str(total_answers))

sample_question_batch_input = build_index_tensor_for_tokenized_sentences(
    tokenized_sentence_list=[instance.question for instance in batch],
    token_to_index_dict=words_to_index_dict,
    index_for_unknown_tokens=num_embeddings - 1).to(device=device)

sample_passage_batch_input = build_index_tensor_for_tokenized_sentences(
    tokenized_sentence_list=[instance.passage for instance in batch],
    token_to_index_dict=words_to_index_dict,
    index_for_unknown_tokens=num_embeddings - 1).to(device=device)

summary_writer.add_graph(qa_module, (sample_question_batch_input, sample_passage_batch_input), True)

summary_writer.close()

# torch.onnx.export(model = qa_module, args = (dummy_question_batch_input, dummy_passage_batch_input), f = "./qa_module3.onnx", verbose = True)
# torch.save(qa_module.state_dict(), "./qa_module")
