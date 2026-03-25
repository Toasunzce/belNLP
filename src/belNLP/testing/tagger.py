import torch
from torch import nn
import torch.nn.functional as F

import sys


# <-------   ------->

class SelfAttention(nn.Module):
    """
    Single-head self-attention module, used to train the following parameters:
    - Q matrix: query vectors
    - K matrix: key vectors
    - 
    """
    def __init__(self, dim: int = 256) -> None:
        super().__init__()

        self.scale = dim ** 0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)

        return torch.matmul(weights, V)
    

class WordEncoder(nn.Module):
    """
    
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden=128, pad_id=0) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden, batch_first=True, bidirectional=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch, words, chars = x.shape           # [batch, words, chars]

        x = x.view(batch * words, chars)        # [batch * words, chars]
        emb = self.embedding(x)                 # [batch * words, chars, emb_size]


        
        _, (h, _) = self.lstm(emb)              # [2, batch * words, hidden_size]
        h = torch.cat([h[0], h[1]], dim=-1)     # [batch * words, 2 * hidden_size]
        h = h.view(batch, words, -1)            # [batch, words, 2 * hidden_size]

        return h
    

class SentenceEncoder(nn.Module):
    """
    
    """
    def __init__(self, input_dim=256, hidden=128) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.attention = SelfAttention()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out, _ = self.lstm(x)                   # [batch, words, 2 * hidden_size]
        out = self.attention(out)
        return out
    

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
# <-------   ------->


def predict(model, sentence: list[str], char2idx, device):
    model.eval()
    max_word_length = max(len(w) for w in sentence)
    padded = []

    for word in sentence:
        encoded = [char2idx.get(ch, char2idx["<UNK>"]) for ch in word]
        encoded += [char2idx["<PAD>"]] * (max_word_length - len(word))
        padded.append(encoded)

    x = torch.tensor([padded], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[0], dim=-1).max(-1)
        preds = logits.argmax(-1)[0]

    return probs


# <-------   ------->
conditions = torch.load("../models/POSTagger.pt", "cpu", weights_only=False)

char2idx = conditions['char2idx']
tag2idx = conditions['tag2idx']
idx2tag = {v: k for k, v in tag2idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = POSTagger(len(char2idx), len(tag2idx) - 1)
model.load_state_dict(conditions['tagger_state_dict'])
model.to(device)

text_split = sys.argv[1].split()
probs, tags = predict(model, text_split, char2idx, device)

print("word           tag      confidence")
for word, prob, tag in zip(text_split, probs, tags):
    print(f"{word:<14} {idx2tag[tag.item()]:<8}   {prob.item():.3f}")


