from typing import List, Tuple, NamedTuple
from collections import namedtuple
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader, default_collate

from src.common.neural_net_param_utils import xavier_normal_weight_init
from src.data.instance.instance import SquadTensorInstance
from src.tokenization.pretrained_bert_tokenizer import PretrainedBertTokenizer
from src.data.dataset.datasetreaders import SquadReader
from src.data.dataset.dataset import SquadDatasetForBert
from src.modules.question_answering_modules import BertQuestionAnsweringModule
from torch.nn.functional import pad
from torch.utils.checkpoint import checkpoint
from pytorch_pretrained_bert.optimization import BertAdam


dataset_data_file_path: str = "../../data/SQuAD"

# training_data_file_path: str = dataset_data_file_path + "/train-v2.0.json"
# dev_data_file_path: str = dataset_data_file_path + "/dev-v2.0.json"

training_data_file_path: str = dataset_data_file_path + "/sample.json"
dev_data_file_path: str = dataset_data_file_path + "/sample.json"


# the number of epochs to train the model for
NUM_EPOCHS: int = 2

# batch size to use for training. The default can be kept small when using gradient accumulation. However, if
# use_gradient_accumulation were to be False, it's best to use a reasonably large batch size
BATCH_SIZE: int = 5

# max-allowed token sequence length for pretrained-model
MAX_ALLOWED_BERT_TOKEN_SEQUENCE_LENGTH: int = 512

use_checkpointing: bool = True
use_gradient_accumulation: bool = True

# num of gradient accumulation steps. Will have no effect if use_gradient_accumulation is false
NUM_GRADIENT_ACCUMULATION_STEPS: int = 10


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SquadTensorTuple: NamedTuple = namedtuple('SquadTensorTuple', ['token_ids', 'segment_ids', 'answer_start_index',
                                                               'answer_end_index', 'attention_mask'])

pretrained_bert_tokenizer = PretrainedBertTokenizer('bert-base-uncased')

answer_start_marker = "π"
answer_end_marker = "ß"

answer_start_marker_with_spaces: str = " %s " % answer_start_marker
answer_end_marker_with_spaces: str = " %s " % answer_end_marker


def collate_with_padding(batch):
    if not isinstance(batch[0], SquadTensorInstance):
        return default_collate(batch)

    max_length_instance = max(batch, key = lambda batch_item: batch_item.token_ids.size()[0])
    max_length = max_length_instance.token_ids.size()[0]
    batch_as_tensor_tuples = []
    for instance in batch:
        instance_length = instance.token_ids.size()[0]
        padding_size = max_length - instance_length
        instance.token_ids = pad(instance.token_ids, (0, padding_size))
        instance.segment_ids = pad(instance.segment_ids, (0, padding_size))
        instance.attention_mask = torch.tensor([1] * instance_length + [0] * padding_size, device = device)
        batch_as_tensor_tuples.append(
            SquadTensorTuple(instance.token_ids, instance.segment_ids, instance.answer_indices[0].item(),
                             instance.answer_indices[1].item(),
                             instance.attention_mask))

    return default_collate(batch_as_tensor_tuples)


def get_squad_dataset_from_file(file_path: str) -> SquadDatasetForBert:
    instances = SquadReader.read(file_path)
    squad_dataset_list: List = []
    num_skipped_instances: int = 0

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

        passage_tokens = pretrained_bert_tokenizer.tokenize(passage_text_for_tokenization)
        question_tokens = pretrained_bert_tokenizer.tokenize(squad_qa_instance_as_dict['question'])

        answer_start_marker_index = passage_tokens.index(answer_start_marker)
        answer_end_marker_index = passage_tokens.index(answer_end_marker)

        # let's remove the answer markers now
        passage_tokens = passage_tokens[:answer_start_marker_index] + \
                         passage_tokens[answer_start_marker_index + 1: answer_end_marker_index] + \
                         passage_tokens[answer_end_marker_index + 1:]

        answer_span_start_token_index: int = answer_start_marker_index + len(question_tokens)

        # removing the start marker, shifts the answer towards the start by an additional index,
        # hence -2 as opposed to -1
        answer_span_end_token_index: int = answer_end_marker_index - 2 + len(question_tokens)

        all_tokens = question_tokens + passage_tokens

        if len(all_tokens) > MAX_ALLOWED_BERT_TOKEN_SEQUENCE_LENGTH:
            num_skipped_instances += 1
            continue

        all_token_ids = pretrained_bert_tokenizer.convert_tokens_to_ids(all_tokens)
        all_segment_ids = [0 for question_token in question_tokens] + [1 for passage_token in passage_tokens]
        answer_indices = (answer_span_start_token_index, answer_span_end_token_index)

        instance_tensor = SquadTensorInstance(torch.tensor(all_token_ids, dtype = torch.long, device = device),
                                              torch.tensor(all_segment_ids, dtype = torch.long, device = device),
                                              torch.tensor(answer_indices, dtype = torch.long, device = device))

        squad_dataset_list.append(instance_tensor)

        # print(passage_tokens)
        # print(question_tokens)
        # print(squad_qa_instance_as_dict['answer'])

    return SquadDatasetForBert(squad_dataset_list), num_skipped_instances


train_dataset, num_skipped_training_instances = get_squad_dataset_from_file(training_data_file_path)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding)

test_dataset, num_skipped_test_instances = get_squad_dataset_from_file(dev_data_file_path)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, collate_fn = collate_with_padding)

print("num skipped training instances: " + str(num_skipped_training_instances))
print("num skipped test instances: " + str(num_skipped_test_instances))

bert_qa_module = BertQuestionAnsweringModule(device = device,
                                             named_param_weight_initializer = xavier_normal_weight_init)
bert_qa_module.to(device)

# this makes sure checkpointing still calculates gradients,
# refer to https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
unused_tensor = torch.zeros((2, 2), requires_grad = True).to(device)

if use_checkpointing:
    # refer to https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    # dropout is not recommended while using checkpointing
    bert_qa_module.bert_model.config.attention_probs_dropout_prob = 0

loss_function = torch.nn.CrossEntropyLoss()
optimizer = BertAdam(bert_qa_module.parameters(), lr = 0.01)

batch_num: int = 0

for epoch in tqdm(range(NUM_EPOCHS)):
    print("inside training loop")
    for data_item in train_dataloader:

        answer_start_index_batch_as_matrix = data_item[2].unsqueeze(dim = 1).to(device = device)
        answer_end_index_batch_as_matrix = data_item[3].unsqueeze(dim = 1).to(device = device)

        if use_checkpointing:
            bert_model_output: Tuple[torch.Tensor, torch.Tensor] = checkpoint(bert_qa_module, data_item[0],
                                                                              data_item[1], data_item[4], unused_tensor)
        else:
            bert_model_output: Tuple[torch.Tensor, torch.Tensor] = bert_qa_module(input_ids = data_item[0],
                                                                                  token_type_ids = data_item[1],
                                                                                  attention_mask = data_item[4],
                                                                                  output_all_encoded_layers = False)

        # Note that pytorch's checkpoint method does not imply the traditional meaning of checkpointing
        # (which generally indicates saving a model/process to be resumed later). Instead, it's a
        # process to avoid memory consumption (caused by reverse mode auto-differentiation) at the cost of performance.
        # More here: https://pytorch.org/docs/stable/checkpoint.html
        # checkpointing was simply added to ensure the model can run on a cpu or low-cost GPUs, else
        # the BERT transformer makes a 12 GB Tesla K80 run out of memory. If you'd like to disable it, you could
        # alternatively use the commented out counterpart which directly calls the bert_qa_module

        batch_num += 1

        answer_start_index_loss = loss_function(bert_model_output[0], answer_start_index_batch_as_matrix)
        answer_end_index_loss = loss_function(bert_model_output[1], answer_end_index_batch_as_matrix)

        total_loss = answer_start_index_loss + answer_end_index_loss

        if use_gradient_accumulation:
            total_loss = total_loss / NUM_GRADIENT_ACCUMULATION_STEPS

        total_loss.backward()

        if not use_gradient_accumulation or batch_num % NUM_GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            bert_qa_module.zero_grad()

        if batch_num % 100 == 0:
            print("output calculation done for batch#: " + str(batch_num))
            print("total loss: " + str(total_loss.data.item()))

        #         print(bert_model_output)
        #         print(bert_model_output[0].size())
        #         print(bert_model_output[1].size())

# eval mode switches off features such as dropout and batch_norm
bert_qa_module.eval()

num_correct_start_index_answers: int = 0
num_correct_end_index_answers: int = 0
total_answers: int = 0

with torch.no_grad():
    for data_item in test_dataloader:
        bert_model_output: Tuple[torch.Tensor, torch.Tensor] = bert_qa_module(input_ids = data_item[0], token_type_ids =
        data_item[1], attention_mask = data_item[4], output_all_encoded_layers = False)

        answer_start_index_batch_as_matrix = data_item[2].unsqueeze(dim = 1).to(device = device)
        answer_end_index_batch_as_matrix = data_item[3].unsqueeze(dim = 1).to(device = device)

        answer_start_index_loss = loss_function(bert_model_output[0], answer_start_index_batch_as_matrix)
        answer_end_index_loss = loss_function(bert_model_output[1], answer_end_index_batch_as_matrix)
        total_loss = answer_start_index_loss + answer_end_index_loss

        answer_start_indices_chosen_by_model = torch.squeeze(torch.topk(bert_model_output[0], 1, dim = 1)[1])
        answer_end_indices_chosen_by_model = torch.squeeze(torch.topk(bert_model_output[1], 1, dim = 1)[1])

        print("batch loss: " + str(total_loss))
        print("answer start indices size: " + str(answer_start_indices_chosen_by_model.size()))
        print("answer end indices size: " + str(answer_end_indices_chosen_by_model.size()))
        print(answer_end_indices_chosen_by_model)

        answer_start_index_comparison_tensor = torch.eq(answer_start_index_batch_as_matrix,
                                                        answer_start_indices_chosen_by_model)
        answer_end_index_comparison_tensor = torch.eq(answer_end_index_batch_as_matrix,
                                                      answer_end_indices_chosen_by_model)

        num_correct_start_index_answers += answer_start_index_comparison_tensor.sum().item()
        num_correct_end_index_answers += answer_end_index_comparison_tensor.sum().item()
        total_answers += answer_start_indices_chosen_by_model.size()[0]

print("start index accuracy: " + str(num_correct_start_index_answers / total_answers))
print("end index accuracy: " + str(num_correct_end_index_answers / total_answers))
print("total answers: " + str(total_answers))