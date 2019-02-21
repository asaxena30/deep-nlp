from typing import List, Iterable

from pytorch_pretrained_bert import BertTokenizer


class PretrainedBertTokenizer:

    def __init__(self, pretrained_bert_model_name = 'bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model_name)

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenize(text) for text in texts]

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)
