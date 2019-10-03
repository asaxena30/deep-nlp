import torch
from typing import List, Tuple


class QAInstanceWithAnswerSpan:
    def __init__(self, question: List[str],  passage: List[str], answer: str, answer_start_and_end_index: Tuple[int, int],
                 total_length: int):
        self.question = question
        self.passage = passage
        self.answer = answer
        self.answer_start_and_end_index = answer_start_and_end_index
        self.total_length = total_length

    def __str__(self):
        return "passage = " + str(self.passage) + ", question = " + str(self.question), + " answer = " + self.answer + \
               ", answer indices = " + str(self.answer_start_and_end_index)


class TaggedQAInstanceWithAnswerSpan(QAInstanceWithAnswerSpan):
    def __init__(self, question: List[str], question_pos_tags: List[str], passage: List[str],
                 passage_pos_tags: List[str], answer: str, answer_start_and_end_index: Tuple[int, int],
                 total_length: int):
        super().__init__(question, passage, answer, answer_start_and_end_index, total_length)
        self.question_pos_tags = question_pos_tags
        self.passage_pos_tags = passage_pos_tags

    def __str__(self):
        return super().__str__() + ", passage_pos_tags = " + str(self.passage_pos_tags) + ", question_pos_tags = " + str(self.question_pos_tags)


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

