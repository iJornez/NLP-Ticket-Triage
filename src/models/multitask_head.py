import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTask(nn.Module):
    def __init__(self, enc: str = "distilbert-base-uncased", n_topic: int = 6, n_sent: int = 3, p: float = 0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(p)
        self.topic = nn.Linear(h, n_topic)
        self.sent = nn.Linear(h, n_sent)

    def forward(self, input_ids, attention_mask):
        x = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.drop(x)
        return self.topic(x), self.sent(x)
