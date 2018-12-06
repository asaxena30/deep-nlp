import io
import torch


def load_fasttext_vectors_as_dict(vectors_file_path):
    data = {}
    with io.open(vectors_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = torch.FloatTensor(list(map(float, tokens[1:])))
    return data

