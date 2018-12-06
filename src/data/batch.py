from typing import List
from abc import ABC
from src.common import CustomTypes


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


class Batch(ABC):

    def __init__(self, items_as_list: List, batch_size: int):
        self.items: List = items_as_list
        self.num_items = len(self.items)
        self.batch_size = batch_size
        super().__init__()

    def iterator(self):
        return BatchIterator(self)

    def get_data_point_at_index(self, index: int):
        return self.items[index]

    def get_num_items(self):
        return self.num_items


class TaggedSentenceBatch(Batch):
    def __init__(self, items_as_list: List[CustomTypes.TaggedSentence], batch_size: int):
        super().__init__(items_as_list, batch_size)
