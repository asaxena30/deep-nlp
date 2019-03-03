from typing import List
import json
from src.errors.custom_errors import UnsupportedOperationError


class Conll2003Reader:
    def __init__(self, filepaths: List[str]):
        super().__init__()
        self.filepaths = filepaths

    def get_tagged_sentences(self) -> List:

        all_sentences_with_tags = []

        for filepath in self.filepaths:
            with open(filepath) as data_file:
                current_sentence = []
                for word_with_tags in data_file:
                    word_with_tags = word_with_tags.strip()

                    if current_sentence and not word_with_tags:
                        all_sentences_with_tags.append(current_sentence)
                        # print(current_sentence)
                        current_sentence = []
                    elif word_with_tags:
                        word_with_tags_splitted = word_with_tags.split()

                        if word_with_tags_splitted[0] != '-DOCSTART-':
                            current_sentence.append(tuple(word_with_tags_splitted))

        return all_sentences_with_tags


class SquadReader:
    """reader class for the Stanford QA dataset. Please refer to  https://rajpurkar.github.io/SQuAD-explorer/
    for details on the structure, goals etc. for this dataset"""

    @staticmethod
    def read(file_path: str, include_unanswerable_questions = False):
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']

            for article in dataset:
                for paragraph_json in article['paragraphs']:
                    paragraph = paragraph_json["context"]

                    for question_answer in paragraph_json['qas']:
                        if include_unanswerable_questions:
                            raise UnsupportedOperationError("no support for unanswerable questions yet")
                        question_text = question_answer["question"].strip().replace("\n", "")
                        answer_texts = [answer['text'] for answer in question_answer['answers']]

                        # effectively, no answers means it's an unanswerable question which isn't supported yet
                        if not answer_texts:
                            continue

                        span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                        span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                        yield {"passage": paragraph,
                               "question": question_text,
                               "answer": answer_texts[0],
                               "span_start": span_starts[0],
                               "span_end": span_ends[0]}
