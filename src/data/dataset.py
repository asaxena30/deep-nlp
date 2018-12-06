from torch.utils.data.dataset import Dataset
from src.data.batch import Batch
from typing import List


class BatchedDataset(Dataset):
    """
        a dataset where each item is an instance of Batch
    """
    def __init__(self, batches: List[Batch]):
        self.batches = batches
        self.num_batches = len(batches)
        super().__init__()

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return self.num_batches

    def iterator(self):
        return BatchedDatasetIterator(self)


class LabeledBatchedDataset(BatchedDataset):
    """
        a batched dataset where each batch-item is a (batch_of_data_points, batch_labels) tuple. in other words
        consider X -> Y where Xs are the examples and Y are labels. This dataset has batches of (X, Y) tuples
    """
    def __init__(self, batches: List[Batch]):
        super().__init__(batches)

    def __getitem__(self, index):
        return self.batches[index][0], self.batches[index][1]


class BatchedDatasetIterator:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index > self.dataset.__len__() - 1:
            raise StopIteration
        return self.dataset.__getitem__(self.current_index)

