import json
import os
from functools import lru_cache
from typing import Dict, List
from transformers import PreTrainedTokenizerFast

@lru_cache()
def default_dict():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")


class CROHMEVocab:
    def __init__(self, dict_path: str = default_dict()) -> None:
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=dict_path)
        with open(dict_path, "rb") as f:
            self.vocab = json.load(f)['model']['vocab']
        self.PAD_IDX = self.vocab['[PAD]']
        self.SOS_IDX = self.vocab['[BOS]']
        self.EOS_IDX = self.vocab['[EOS]']

    def words2indices(self, words: List[str]) -> List[int]:
        return self.tokenizer(words[0])['input_ids']

    def indices2words(self, id_list: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(id_list)

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.vocab)


if __name__ == '__main__':
    tex = r"\pi_{1}(T,t_{0})\approx\pi_{1}(S^{1},x_{0})\times\pi_{1}(S^{1},y_{0})\cong\mathbf{Z}\times\mathbf{Z}=\mathbf{Z}^{2}."
    x = CROHMEVocab()
    print(x.words2indices([tex]))
