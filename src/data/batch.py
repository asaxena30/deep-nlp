import torch
from typing import List, Callable, Any

from torch import Tensor

from src.common import CustomTypes



class Batch:

    def __init__(self, items_as_list: List):
        super().__init__()
        self.items: List = items_as_list
        self.num_items = len(self.items)

    def iterator(self):
        return BatchIterator(self)

    def get_data_point_at_index(self, index: int):
        return self.items[index]

    def get_num_items(self):
        return self.num_items

    def size(self):
        """ alias for get_num_items"""
        return self.get_num_items()

    def map(self, mapper: Callable[[List], Any]):
        return mapper(self.items)


class TensorBatch(Batch):
    def __init__(self, tensor_list: List[Tensor], use_padding):
        super().__init__(tensor_list)

    def as_tensor(self):
        return torch.Tensor(self.items)


class BatchIterator:

    def __init__(self, batch: Batch):
        self.start = 0
        self.current_index = self.start
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index > self.batch.get_num_items() - 1:
            raise StopIteration
        next_data_point = self.batch.get_data_point_at_index(self.current_index)
        self.current_index += 1
        return next_data_point


class TaggedSentenceBatch(Batch):
    def __init__(self, items_as_list: List[CustomTypes.TaggedSentence]):
        super().__init__(items_as_list)
