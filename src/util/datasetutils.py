import io
import torch
from collections import OrderedDict


def load_word_vectors_as_ordered_dict(vectors_file_path, expected_embedding_size: int = None):
    word_to_vector_dict = OrderedDict()
    with io.open(vectors_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        for line in fin:
            tokens = line.rstrip().split(' ')
            embedding_tensor = torch.FloatTensor(list(map(float, tokens[1:])))

            if expected_embedding_size is None or embedding_tensor.shape[0] == expected_embedding_size:
                word_to_vector_dict[tokens[0]] = embedding_tensor
    return word_to_vector_dict

