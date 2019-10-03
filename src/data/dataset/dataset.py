import json
from torch.utils.data.dataset import Dataset
from src.data.batch import Batch
from typing import List, NamedTuple

from src.data.instance.instance import QAInstanceWithAnswerSpan, TaggedQAInstanceWithAnswerSpan


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

    def __iter__(self):
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

    def __next__(self):
        if self.current_index > self.dataset.__len__() - 1:
            raise StopIteration
        return self.dataset.__getitem__(self.current_index)


class SquadDatasetForBert(Dataset):
    """
        instances: a list of tuples such that each tuple is an instance containing
        tensors for
            1. (question + passage) tokens Tensor
            2. segment_ids Tensor
            2. answer start index and end-index as tensor
    """
    def __init__(self, instances: List[NamedTuple]):
        self.instances = instances

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)


class SquadDataset(Dataset):
    """
        instances: a list of SquadInstances
    """
    def __init__(self, instances: List[QAInstanceWithAnswerSpan]):
        self.instances = instances

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)


class SquadDatasetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SquadDataset) or isinstance(obj, QAInstanceWithAnswerSpan):
            attributes_dict = {"__type__": type(obj).__name__}
            attributes_dict.update(obj.__dict__)
            return attributes_dict
        return json.JSONEncoder.default(self, obj)


class SquadDatasetDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook = self.dict_to_object)

    def dict_to_object(self, obj):
        if '__type__' not in obj:
            return obj

        obj_type = obj['__type__']

        if obj_type == 'SquadDataset':
            return SquadDataset(obj['instances'])
        elif obj_type == 'QAInstanceWithAnswerSpan':
            return QAInstanceWithAnswerSpan(obj['question'], obj['passage'], obj['answer'],
                                            obj['answer_start_and_end_index'], obj['total_length'])
        elif obj_type == 'TaggedQAInstanceWithAnswerSpan':
            return TaggedQAInstanceWithAnswerSpan(obj['question'], obj['question_pos_tags'], obj['passage'],
                                                  obj['passage_pos_tags'], obj['answer'],
                                                  obj['answer_start_and_end_index'], obj['total_length'])
        else:
            raise NotImplementedError("no deserialization support available for type: " + obj_type)
