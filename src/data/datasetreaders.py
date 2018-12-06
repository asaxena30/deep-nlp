from typing import List


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
