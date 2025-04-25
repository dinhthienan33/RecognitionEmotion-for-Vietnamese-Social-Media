import json
import torch
from typing import List

class Vocab:
    def __init__(self, train_path: str, dev_path: str, test_path: str):
        self.initialize_special_tokens()
        self.make_vocab(train_path, dev_path, test_path)
        self.total_labels = len(self.l2i) 
    def initialize_special_tokens(self) -> None:
        self.cls_token = "<cls>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.specials = [self.pad_token, self.cls_token, self.unk_token]

        self.pad_idx = 0
        self.cls_idx = 1
        self.unk_idx = 2

    def make_vocab(self, train_path: str, dev_path: str, test_path: str):
        self.stoi = {token: idx for idx, token in enumerate(self.specials)}
        self.itos = {idx: token for idx, token in enumerate(self.specials)}
        self.l2i = {}
        self.i2l = {}

        for path in [train_path, dev_path, test_path]:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.values():
                    self.add_sentence_to_vocab(item['review'])
                    self.add_label_to_vocab(item['label'])

    def add_sentence_to_vocab(self, sentence: str):
        tokens = sentence.strip().split()
        for token in tokens:
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def add_label_to_vocab(self, label: str):
        if label not in self.l2i:
            idx = len(self.l2i)
            self.l2i[label] = idx
            self.i2l[idx] = label

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        tokens = sentence.strip().split()
        indices = [self.stoi.get(token, self.unk_idx) for token in tokens]
        return torch.tensor(indices).long()

    def decode_sentence(self, indices: torch.Tensor) -> str:
        tokens = [self.itos[idx] for idx in indices.tolist()]
        return " ".join(tokens)

    def encode_label(self, label: str) -> torch.Tensor:
        return torch.tensor([self.l2i[label]]).long()

    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        labels = label_vecs.tolist()
        return [self.i2l[label] for label in labels]

    def __eq__(self, other):
        return self.stoi == other.stoi and self.itos == other.itos

    def __len__(self):
        return len(self.stoi)
