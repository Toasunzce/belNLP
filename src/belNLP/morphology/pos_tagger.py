import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing_extensions import Self


from belNLP.morphology.base import BaseAnnotator, MorphToken


class _SelfAttention(nn.Module):
    """
    Single-head self-attention module, used to train the following parameters:
    - Q matrix: query vectors
    - K matrix: key vectors
    - V matrix: value vectors
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



class _WordEncoder(nn.Module):
    """
    Character-level BiLSTM encoder.
    Encodes each word as a sequence of characters.
    Input:  [batch, words, chars]
    Output: [batch, words, hidden*2]
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden=128, pad_id=0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, words, chars = x.shape           # [batch, words, chars]
        x = x.view(batch * words, chars)        # [batch * words, chars]
        emb = self.embedding(x)                 # [batch * words, chars, emb_size]
        _, (h, _) = self.lstm(emb)              # [2, batch * words, hidden_size]
        h = torch.cat([h[0], h[1]], dim=-1)     # [batch * words, 2 * hidden_size]
        h = self.dropout(h)
        h = h.view(batch, words, -1)            # [batch, words, 2 * hidden_size]
        return h



class _SentenceEncoder(nn.Module):
    """
    Word-level BiLSTM + self-attention encoder.
    Contextualizes word representations at sentence level.
    Input:  [batch, words, hidden*2]
    Output: [batch, words, hidden*2]
    """
    def __init__(self, input_dim=256, hidden=128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.attention = _SelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                   # [batch, words, 2 * hidden_size]
        out = self.attention(out)
        return out



class _POSTaggerModel(nn.Module):
    """
    Full POS tagging neural network.
    CharBiLSTM -> SentenceBiLSTM + Attention -> Linear
    """

    def __init__(
            self,
            vocab_size,
            num_tags,
            embedding_dim=64,
            word_hidden=128,
            sent_hidden=128,
            pad_id=0,
    ) -> None:
        super().__init__()
        self.word_encoder = _WordEncoder(
            vocab_size, embedding_dim, word_hidden, pad_id
        ) # [word1vec, word2vec, word3vec]
        self.sent_encoder = _SentenceEncoder(
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



class POSTagger(BaseAnnotator[str, MorphToken]):
    """
    Part-of-speech tagger for Belarusian text.
    Trained on UD Belarusian HSE corpus.
    Tags: NOUN, VERB, ADJ, ADV, PRON, DET,
          ADP, CONJ, PART, INTJ, NUM, PUNCT, X

    Usage:
        ```
        >>> tagger = POSTagger.load("models/POSTagger.pt")
        >>> result = tagger.annotate(["Я", "іду", "дадому"])
        >>> assert result[1].pos == "VERB"
        ```
    """
    def __init__(self, model: _POSTaggerModel,
                 char2idx: dict[str, int],
                 idx2tag: dict[int, str],
                 device: torch.device):
        self._model = model
        self._char2idx = char2idx
        self._idx2tag = idx2tag
        self._device = device

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        char2idx = checkpoint["char2idx"]
        tag2idx  = checkpoint["tag2idx"]
        idx2tag  = {v: k for k, v in tag2idx.items()}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _POSTaggerModel(
            vocab_size=len(char2idx),
            num_tags=len(tag2idx) - 1       # excluding <PAD>
        )
        model.load_state_dict(checkpoint["tagger_state_dict"])
        model.to(device)
        model.eval()

        return cls(model, char2idx, idx2tag, device)

    def _encode(self, tokens: list[str]) -> torch.Tensor:
        """

        """
        max_word_len = max(len(w) for w in tokens)
        padded = []

        for word in tokens:
            encoded = [
                self._char2idx.get(ch, self._char2idx["<UNK>"])
                for ch in word
            ]
            encoded += [self._char2idx["<PAD>"]] * (max_word_len - len(word))
            padded.append(encoded)

        return torch.tensor([padded], dtype=torch.long).to(self._device)


    @torch.no_grad()
    def annotate(self, tokens: list[str]) -> list[MorphToken]:
        """

        """
        self._model.eval()

        x = self._encode(tokens)
        logits = self._model(x)             # [1, words, num_tags]
        preds  = logits.argmax(-1)[0]       # [words]
        probs  = F.softmax(logits[0], dim=-1).max(-1).values  # [words]

        result = []
        for token, pred, prob in zip(tokens, preds, probs):
            result.append(MorphToken(
                text=token,
                pos=self._idx2tag[pred.item()],
                morph={"pos_prob": str(round(prob.item(), 4))},
            ))

        return result
    
    def __call__(self, tokens: list[str]) -> list[MorphToken]:
        return self.annotate(tokens)