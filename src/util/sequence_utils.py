from typing import List, Iterable
from itertools import groupby


def build_uniformly_sized_batches(max_batch_size: int, elements: List) -> List:
    sorted_elements: List = sorted(elements, key=len)
    elements_grouped_by_length = groupby(sorted_elements, key=len)
    batches: List = []

    for length, grouper in elements_grouped_by_length:
        if length > 0:
            batches.extend(split_if_longer_than_given_size(grouper, max_batch_size))

    return batches


def split_if_longer_than_given_size(iterable: Iterable, max_iterable_size: int) -> List:

    iterable_list = list(iterable)

    if len(iterable_list) <= max_iterable_size:
        return [iterable_list]

    return [iterable_list[i:i + max_iterable_size] for i in range(0, len(iterable_list), max_iterable_size)]

