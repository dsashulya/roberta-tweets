from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn
from typing import Tuple

BATCH_SIZE = 64
MAX_LEN = 128
DEVICE = "cpu"


class RobertaClassifier(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int, dropout: float):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.fc1 = nn.Linear(768, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.roberta(input_ids, attention_mask)
        z = self.fc1(z.pooler_output)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z

    def compute_probabilities(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.forward(input_ids, attention_mask)
        return nn.Softmax(dim=1)(z)


def tokenize(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    encodings = tokenizer(
        text,
        return_token_type_ids=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors="pt")
    return encodings['input_ids'], encodings['attention_mask']
