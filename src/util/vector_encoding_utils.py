import torch
from src.util import datasetutils, sequence_utils
from typing import Dict, List, Tuple
from src.common import CustomTypes


def load_fasttext():
    fasttext_file_path: str = "/Users/asaxena/Downloads/fasttext_300d/fasttext-300d.vec"
    fasttext: Dict[str, torch.Tensor] = datasetutils.load_word_vectors_as_ordered_dict(fasttext_file_path,
                                                                                       expected_embedding_size = 300)
    return fasttext


def build_one_hot_encoded_tensors_for_tags(tags: List[str]) -> Dict[str, torch.Tensor]:
    tags_to_tensor_rep_dict = {}
    num_tags = len(tags)

    for i in range(num_tags):
        tag_as_tensor = torch.zeros(num_tags)
        tag_as_tensor[i] = 1
        tags_to_tensor_rep_dict[tags[i]] = tag_as_tensor

    return tags_to_tensor_rep_dict


def get_fasttext_word_vector(word: str, fasttext) -> CustomTypes.TorchTensor:
    try:
        return fasttext[word]
    except KeyError:
        return torch.zeros(300)


def get_word_embeddings_for_sentence(sentence: List, fasttext, has_tags: bool=False) -> List[CustomTypes.TorchTensor]:
    return [get_fasttext_word_vector(word_with_tag[0].lower(), fasttext) for word_with_tag in sentence] if has_tags \
        else [get_fasttext_word_vector(word.lower(), fasttext) for word in sentence]


def get_word_embeddings_for_test_sentence(sentence: List) -> List[CustomTypes.TorchTensor]:
    fasttext = load_fasttext()
    return [get_fasttext_word_vector(word.lower(), fasttext) for word in sentence]


def get_word_embeddings_for_tagged_sentences(training_sentences_with_tags: List[List[Tuple]], label_index_in_tuples: int,
                                             batch_size: int, label_tags: List[str]):

    fasttext = load_fasttext()
    tuples_of_sentence_batches_with_labels = sequence_utils.build_uniformly_sized_batches(batch_size, training_sentences_with_tags)
    all_sentence_batches_as_tensor_tuples: List = []
    tag_to_tensor_dict: Dict = build_one_hot_encoded_tensors_for_tags(label_tags)

    for tagged_sentence_batch_with_labels in tuples_of_sentence_batches_with_labels:
        embeddings_lists_for_sentences_in_batch: List = []
        labels_lists_for_sentences_in_batch: List = []
        for tagged_sentence in tagged_sentence_batch_with_labels:

            word_embedding_tensors_list = get_word_embeddings_for_sentence([word[0] for word in tagged_sentence], fasttext)
            word_label_tensors_list = [tag_to_tensor_dict[word[label_index_in_tuples]] for word in tagged_sentence]
            # print(len(word_embedding_tensors_list))
            # print(word_embedding_tensors_list[0].shape)
            # print(len(word_label_tensors_list))
            # print(word_label_tensors_list[0].shape)
            sentence_embedding = torch.cat(word_embedding_tensors_list, 0).view(len(tagged_sentence), 1, -1)
            sentence_labels = torch.cat(word_label_tensors_list, 0).view(len(tagged_sentence), 1, -1)
            embeddings_lists_for_sentences_in_batch.append(sentence_embedding)
            labels_lists_for_sentences_in_batch.append(sentence_labels)
        sentence_batch_embedding_tensor = torch.cat(embeddings_lists_for_sentences_in_batch, 1)
        sentence_batch_labels_tensor = torch.cat(labels_lists_for_sentences_in_batch, 1)
        all_sentence_batches_as_tensor_tuples.append((sentence_batch_embedding_tensor, sentence_batch_labels_tensor))

    return all_sentence_batches_as_tensor_tuples


def build_index_tensor_for_tokenized_sentences(tokenized_sentence_list: List[List[str]], token_to_index_dict: Dict,
                                               index_for_unknown_tokens: int):
    sentence_tensor_list: List[torch.Tensor] = []
    for tokenized_sentence in tokenized_sentence_list:
        sentence_tensor = torch.tensor([token_to_index_dict[token] if token in token_to_index_dict else index_for_unknown_tokens
                                        for token in tokenized_sentence], dtype = torch.long)
        sentence_tensor_list.append(sentence_tensor)

    return torch.stack(sentence_tensor_list, dim = 0)
