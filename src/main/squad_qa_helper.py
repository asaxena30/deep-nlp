import pickle
from typing import Tuple, Dict, Union

import torch
from torch.nn import Module

from src.data.dataset.dataset import SquadDataset
from src.data.instance.instance import QAInstanceWithAnswerSpan
from src.modules.question_answering_modules import QAModuleOutputs
from src.util.vector_encoding_utils import build_index_tensor_for_tokenized_sentences

padding_token: str = '<pad>'


def get_qa_model_outputs(batch, words_to_index_dict: Dict, num_embeddings: int, device: Union[torch.device, str, None],
                         qa_module: Module, iteration_num: int = None) -> QAModuleOutputs:

    question_batch_index_tensor: torch.Tensor = build_index_tensor_for_tokenized_sentences(
        tokenized_sentence_list=[instance.question for instance in batch],
        token_to_index_dict=words_to_index_dict,
        index_for_unknown_tokens=num_embeddings - 1).to(device=device)

    passage_batch_index_tensor: torch.Tensor = build_index_tensor_for_tokenized_sentences(
        tokenized_sentence_list=[instance.passage for instance in batch],
        token_to_index_dict=words_to_index_dict,
        index_for_unknown_tokens=num_embeddings - 1).to(device=device)

    outputs = qa_module(question_batch_index_tensor, passage_batch_index_tensor, iteration_num)

    if type(outputs) == QAModuleOutputs:
        return outputs
    else:
        if type(outputs) != tuple:
            raise NotImplementedError(f"output type of {str(type(outputs))} isn't supported")
        else:
            if len(outputs) not in (2, 3):
                raise NotImplementedError(f"output length of {str(len(outputs))} isn't supported yet")

            return QAModuleOutputs(outputs[0], outputs[1], outputs[2] if len(outputs) == 3 else None)


def get_start_and_end_index_losses_and_original_indices(batch, start_index_outputs, end_index_outputs, loss_function,
                                                        device, return_original_indices: bool = True) -> Tuple:

    batch_start_and_end_indices = [instance.answer_start_and_end_index for instance in batch]

    answer_start_and_end_indices_original = torch.tensor(batch_start_and_end_indices, dtype=torch.long)

    answer_start_indices_original, answer_end_indices_original = torch.chunk(answer_start_and_end_indices_original,
                                                                             chunks=2, dim=1)

    start_index_loss = loss_function(start_index_outputs, answer_start_indices_original.to(device=device))
    end_index_loss = loss_function(end_index_outputs, answer_end_indices_original.to(device=device))

    return start_index_loss, end_index_loss, answer_start_indices_original, answer_end_indices_original \
        if return_original_indices else start_index_loss, end_index_loss


def get_total_loss_and_original_indices(batch, start_index_outputs, end_index_outputs, loss_function,
                                        device, return_original_indices: bool = True) -> Tuple:
    losses_and_original_indices = get_start_and_end_index_losses_and_original_indices(batch=batch,
                                                                                      start_index_outputs=start_index_outputs,
                                                                                      end_index_outputs=end_index_outputs,
                                                                                      loss_function=loss_function,
                                                                                      device=device,
                                                                                      return_original_indices=return_original_indices)

    total_loss = losses_and_original_indices[0] + losses_and_original_indices[1]

    return total_loss, losses_and_original_indices[2], losses_and_original_indices[
        3] if return_original_indices else total_loss


def collate_with_padding(batch):
    if not isinstance(batch[0], QAInstanceWithAnswerSpan):
        raise NotImplementedError("only a QAInstanceWithAnswerSpan/subclass is supported")

    max_question_length_instance = max(batch, key=lambda batch_item: len(batch_item.question))
    max_question_length = len(max_question_length_instance.question)

    max_passage_length_instance = max(batch, key=lambda batch_item: len(batch_item.passage))
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


def calculate_loss_scaling_factor(original_start_indices, original_end_indices,
                                  model_start_index_scores, model_end_index_scores):
    start_indices_from_model = torch.max(model_start_index_scores, dim=1)[1]
    end_indices_from_model = torch.max(model_end_index_scores, dim=1)[1]

    lower_index_max = torch.max(original_start_indices, start_indices_from_model)
    upper_index_min = torch.min(original_end_indices, end_indices_from_model)

    # note: we need this code to work with pytorch 1.0 which is why the older version of
    # clamp is being used which forces specifying both a min and max. The max value here is artificial and specific
    # to this dataset
    overlap_range_length_tensor = torch.clamp(upper_index_min - lower_index_max + 1, min=0, max=10000)

    original_answer_length_tensor = original_end_indices - original_start_indices + 1
    model_answer_length_tensor = torch.abs(end_indices_from_model - start_indices_from_model) + 1

    # basically 1 - 2 * overlap_range_length/(actual_answer_length + model_answer_length)
    return 1 - torch.mean(torch.div(2 * overlap_range_length_tensor.float(),
                                    (original_answer_length_tensor + model_answer_length_tensor).float()))


def load_serialized_dataset(datafile_path: str) -> SquadDataset:
    with open(datafile_path, "rb") as f:
        return pickle.load(f)
