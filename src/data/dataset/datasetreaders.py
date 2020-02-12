from typing import List

from nltk import word_tokenize, pos_tag

from src.data.dataset.dataset import SquadDataset
from src.data.instance.instance import QAInstanceWithAnswerSpan, TaggedQAInstanceWithAnswerSpan
from src.errors.custom_errors import UnsupportedOperationError
import rapidjson


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

    answer_start_marker = "π"
    answer_end_marker = "ß"

    answer_start_marker_with_spaces: str = " %s " % answer_start_marker
    answer_end_marker_with_spaces: str = " %s " % answer_end_marker

    @staticmethod
    def read(file_path: str, include_unanswerable_questions = False):
        with open(file_path) as dataset_file:
            dataset = SquadReader.extract_dataset(dataset_file)

            for article in dataset:
                for paragraph_json in article['paragraphs']:
                    paragraph = SquadReader.correct_known_typos(paragraph_json["context"])

                    for question_answer in paragraph_json['qas']:
                        if include_unanswerable_questions:
                            raise UnsupportedOperationError("no support for unanswerable questions yet")
                        question_text = SquadReader.correct_known_typos(question_answer["question"].strip().replace("\n", ""))
                        all_answers = sorted(question_answer['answers'], key = lambda ans: len(ans['text']))
                        answer_texts = [answer['text'] for answer in all_answers]

                        # effectively, no answers means it's an unanswerable question which isn't supported yet
                        if not answer_texts:
                            continue

                        span_starts = [answer['answer_start'] for answer in all_answers]
                        span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                        yield {"id": question_answer["id"],
                               "passage": paragraph,
                               "question": question_text,
                               "answer": answer_texts[0],
                               "span_start": span_starts[0],
                               "span_end": span_ends[0]}

    @staticmethod
    def extract_dataset(dataset_file):
        dataset_json = rapidjson.load(dataset_file)
        return dataset_json['data']

    @staticmethod
    def correct_known_typos(question_or_passage_text: str):
        return question_or_passage_text.replace("assimilted", "assimilated")

    @staticmethod
    def get_squad_dataset_from_file(file_path: str, add_pos_tags: bool = False) -> SquadDataset:
        instances = SquadReader.read(file_path)
        squad_instance_list: List = []

        for squad_qa_instance_as_dict in instances:
            if 'span_start' not in squad_qa_instance_as_dict:
                continue

            qa_id: str = squad_qa_instance_as_dict["id"]
            span_start_char_index: int = squad_qa_instance_as_dict['span_start']
            span_end_char_index: int = squad_qa_instance_as_dict['span_end']

            passage_text = squad_qa_instance_as_dict['passage']

            # here we are simply inserting answer start and end markers so that after tokenization we can still track
            # the answer boundaries
            passage_text_for_tokenization = passage_text[:span_start_char_index] + SquadReader.answer_start_marker_with_spaces + \
                                            passage_text[
                                            span_start_char_index: span_end_char_index] + SquadReader.answer_end_marker_with_spaces + \
                                            passage_text[span_end_char_index:]

            passage_tokens = word_tokenize(passage_text_for_tokenization)
            question_tokens = word_tokenize(squad_qa_instance_as_dict['question'])

            answer_start_marker_index = passage_tokens.index(SquadReader.answer_start_marker)
            answer_end_marker_index = passage_tokens.index(SquadReader.answer_end_marker)

            # let's remove the answer markers now
            passage_tokens = passage_tokens[:answer_start_marker_index] + \
                             passage_tokens[answer_start_marker_index + 1: answer_end_marker_index] + \
                             passage_tokens[answer_end_marker_index + 1:]

            answer_span_start_token_index: int = answer_start_marker_index

            # removing the start marker, shifts the answer towards the start by an additional index,
            # hence -2 as opposed to -1
            answer_span_end_token_index: int = answer_end_marker_index - 2

            answer_indices = (answer_span_start_token_index, answer_span_end_token_index)

            if add_pos_tags:
                instance = TaggedQAInstanceWithAnswerSpan(question = question_tokens,
                                                    question_pos_tags = pos_tag(question_tokens),
                                                    passage = passage_tokens,
                                                    passage_pos_tags = pos_tag(passage_tokens),
                                                    answer = squad_qa_instance_as_dict['answer'],
                                                    answer_start_and_end_index = answer_indices,
                                                    total_length = len(question_tokens + passage_tokens),
                                                    id = qa_id)
            else:
                instance = QAInstanceWithAnswerSpan(question = question_tokens,
                                                    passage = passage_tokens,
                                                    answer = squad_qa_instance_as_dict['answer'],
                                                    answer_start_and_end_index = answer_indices,
                                                    total_length = len(question_tokens + passage_tokens),
                                                    id = qa_id)

            squad_instance_list.append(instance)

            # print(passage_tokens)
            # print(question_tokens)
            # print(squad_qa_instance_as_dict['answer'])

        return SquadDataset(sorted(squad_instance_list, key = lambda x: x.total_length))
