import torch
from typing import List, Tuple


class QAInstanceWithAnswerSpan:
    def __init__(self, question: List[str], passage: List[str], answer: str, answer_start_and_end_index: Tuple[int, int],
                 total_length: int):
        self.question = question
        self.passage = passage
        self.answer = answer
        self.answer_start_and_end_index = answer_start_and_end_index
        self.total_length = total_length


class QATensorInstanceWithAnswerSpan:
    def __init__(self, question: torch.Tensor, passage: torch.Tensor, answer_start_and_end_index: Tuple[int, int],
                 answer: torch.Tensor = None):
        self.question = question
        self.passage = passage
        self.answer = answer
        self.answer_start_and_end_index = answer_start_and_end_index


class SquadTensorInstanceForBert:
    def __init__(self, token_ids: torch.Tensor, segment_ids: torch.Tensor, answer_indices: torch.Tensor):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.answer_indices = answer_indices


class SquadInstanceForBert(QAInstanceWithAnswerSpan):
    def __init__(self, token_ids: List[int], segment_ids: List[int],
                 question: str, passage: str, answer: str, answer_start_and_end_index: Tuple[int, int]):
        super().__init__(question, passage, answer, answer_start_and_end_index)
        self.token_ids = token_ids
        self.segment_ids = segment_ids

    def token_ids_as_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.token_ids)

    def segment_ids_as_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.segment_ids)

