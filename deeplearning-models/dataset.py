import torch
from torch.utils.data import Dataset
from typing import List, Dict
import json
from vocab import Vocab

def collate_fn(items: List[Dict]) -> dict:

    def pad_value(input: torch.Tensor, value: int, max_length: int) -> torch.Tensor:
        if input.shape[-1] < max_length:
            delta_len = max_length - input.shape[-1]
            pad_tensor = torch.tensor([value] * delta_len).long()
            input = torch.cat([input, pad_tensor], dim=-1)
        return input

    max_len = max(item["input_ids"].shape[-1] for item in items)
    batch_input_ids = []
    batch_labels = []
    
    for item in items:
        # Pad input IDs
        input_ids = item["input_ids"]
        input_ids = pad_value(input_ids, value=0, max_length=max_len)
        batch_input_ids.append(input_ids.unsqueeze(0))

        # Pad labels
        labels = item["label"]
        labels = pad_value(labels, value=-100, max_length=max_len)  # -100 for CrossEntropyLoss
        batch_labels.append(labels.unsqueeze(0))

    return {
        "input_ids": torch.cat(batch_input_ids),
        "labels": torch.cat(batch_labels)
    }

class ViSMEC(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        with open(path, encoding='utf-8') as file:
            _data = json.load(file)
        self._data = list(_data.values())
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        item = self._data[index]
        
        sentence = item["review"]
        label = item["label"]

        # Encoding sentence and label using Vocab class
        encoded_sentence = self._vocab.encode_sentence(sentence)  # Expected: Tensor [seq_len]
        encoded_label = self._vocab.encode_label(label)           # Expected: Tensor [seq_len]
      
        return {
            "input_ids": encoded_sentence,
            "label": encoded_label
        }
