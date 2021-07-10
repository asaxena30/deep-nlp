from typing import Iterable, Dict


class Vocab:
    def __init__(self, tokens: Iterable[str] = None):
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}

        if tokens:
            self.add_tokens(tokens=tokens)

    def __call__(self, token: str):
        if token not in self.word_to_index:
            return self.word_to_index['<unk>']
        return self.word_to_index[token]

    def __len__(self):
        return len(self.word_to_index)

    def add_token(self, token: str):
        next_index: int = len(self.word_to_index)
        self.word_to_index[token] = next_index
        self.index_to_word[next_index] = token

    def add_tokens(self, tokens: Iterable[str]):
        next_index: int = len(self.word_to_index)
        for token in tokens:
            self.word_to_index[token] = next_index
            self.index_to_word[next_index] = token
            next_index += 1
