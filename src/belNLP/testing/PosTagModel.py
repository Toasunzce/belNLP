import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn





class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        ...

    def forward(self):
        ...


class SelfAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.scale = torch.sqrt(dim)

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Q = self.query
        K = self.key
        V = self.value

        weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / scale)

        return torch.matmul(weights, V)
    

class CharEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim=32, hidden=128) -> None:

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        _, (h, _) = self.lstm(x)
        

class POSTagger(nn.Module):
    """
    
    """
    def __init__(
            self,
            vocab_size,
            num_tags,
            embedding_dim=128,
            word_hidden=128,
            sent_hidden=128,
            pad_id=0,
    ) -> None:
        super().__init__()

        self.word_encoder = WordEncoder(
            vocab_size, embedding_dim, word_hidden, pad_id
        ) # [word1vec, word2vec, word3vec]

        self.sent_encoder = SentenceEncoder(
            2 * word_hidden, sent_hidden
        ) # [word1|conеxt, word2|context, word3|context]
        self.linear = nn.Linear(
            2 * sent_hidden, num_tags
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        word_repr = self.word_encoder(x)
        word_repr_context = self.sent_encoder(word_repr)
        logits = self.linear(word_repr_context)

        return logits