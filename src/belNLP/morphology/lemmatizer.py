from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from belNLP.morphology.base import BaseAnnotator, MorphToken



PAD_ID = 0
BOS_ID = 1
EOS_ID = 2



class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])  # ty:ignore[not-subscriptable]



class _LemmatizerModel(nn.Module):
    """
    Character-level encoder-decoder transformer.
    """
 
    def __init__(
        self,
        vocab_size:      int,
        pos_size:        int,
        d_model:         int   = 128,
        nhead:           int   = 4,
        num_enc_layers:  int   = 3,
        num_dec_layers:  int   = 3,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
        max_len:         int   = 256,
        pad_id:          int   = PAD_ID,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id  = pad_id
 
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(pos_size, d_model)
        self.pos_enc   = _PositionalEncoding(d_model, dropout, max_len)
 
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_enc_layers)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_dec_layers)
        self.input_proj = nn.Linear(d_model * 2, d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights() # FIXME remove this shi???
 
    # useless af?????
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
    def encode(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        pad_mask = src.eq(self.pad_id)

        scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.src_embed(src) * scale

        pos_emb = self.pos_embed(pos).unsqueeze(1).expand(-1, src.size(1), -1)
        x = torch.cat([x, pos_emb], dim=-1)

        x = self.input_proj(x)
        x = self.pos_enc(x)

        return self.encoder(x, src_key_padding_mask=pad_mask)
 
    def decode_step(self, tgt: torch.Tensor, # tgt [B, tgt_len] -> logits [B, tgt_len, vocab]
                    memory: torch.Tensor,
                    src: torch.Tensor) -> torch.Tensor:
        
        tgt_len  = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        y = self.pos_enc(self.tgt_embed(tgt) * scale)
        out = self.decoder(
            y, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt.eq(self.pad_id),
            memory_key_padding_mask=src.eq(self.pad_id),
        )
        return self.out_proj(out)                                    # [B, tgt_len, vocab]
 
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src, pos)
        return self.decode_step(tgt[:, :-1], memory, src)           # [B, tgt_len-1, vocab]
    


@torch.no_grad()
def beam_search(
    model:          _LemmatizerModel,
    src:            torch.Tensor,    # [1, src_len]
    pos:            torch.Tensor,    # [1]
    beam_size:      int   = 5,
    max_len:        int   = 32,
    length_penalty: float = 0.6,
) -> list[int]:
    model.eval()
    device = src.device
    memory = model.encode(src, pos)                                  # [1, src_len, d]
 
    # (score, tokens)
    beams:     list[tuple[float, list[int]]] = [(0.0, [BOS_ID])]
    completed: list[tuple[float, list[int]]] = []
 
    for _ in range(max_len):
        if not beams:
            break
        candidates: list[tuple[float, list[int]]] = []
 
        for score, tokens in beams:
            tgt    = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode_step(tgt, memory, src)            # [1, len, vocab]
            lp     = F.log_softmax(logits[0, -1], dim=-1)           # [vocab]
 
            topk_lp, topk_ids = lp.topk(beam_size)
            for lp_val, idx in zip(topk_lp.tolist(), topk_ids.tolist()):
                new_seq   = tokens + [idx]
                new_score = score + lp_val
                if idx == EOS_ID:
                    norm = ((5 + len(new_seq)) / 6) ** length_penalty
                    completed.append((new_score / norm, new_seq))
                else:
                    candidates.append((new_score, new_seq))
 
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]
 
    if not completed:
        completed = beams  # type: ignore[assignment]
 
    completed.sort(key=lambda x: x[0], reverse=True)
    return completed[0][1]
 

 
def ids_to_str(tokens: list[int], idx2char: dict[int, str]) -> str:
    result = []
    for i in tokens:
        if i in (BOS_ID, PAD_ID):
            continue
        if i == EOS_ID:
            break
        result.append(idx2char.get(i, ""))
    return "".join(result)



class Lemmatizer(BaseAnnotator[MorphToken, MorphToken]):
    def __init__(self,
                 model: _LemmatizerModel,
                 char2idx: dict[str, int],
                 idx2char: dict[int, str],
                 pos2idx: dict[str, int],
                 device: torch.device):
        self._model = model
        self._char2idx = char2idx
        self._idx2char = idx2char
        self._pos2idx = pos2idx
        self._device = device

    @classmethod
    def load(cls, path: str | Path) -> Self:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]
        pos2idx  = checkpoint["pos2idx"]
        state    = checkpoint["model_state_dict"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = checkpoint.get("model_config", {})

        model = _LemmatizerModel(
            vocab_size=cfg.get("vocab_size", len(char2idx)),
            pos_size=cfg.get("pos_size", len(pos2idx)),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_enc_layers=cfg.get("num_enc_layers", 3),
            num_dec_layers=cfg.get("num_dec_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )

        model.load_state_dict(state)
        model.to(device)
        model.eval()

        return cls(model, char2idx, idx2char, pos2idx, device)

    def _encode_word(self, word: str) -> torch.Tensor:
        unk = self._char2idx["<UNK>"]
        ids = [self._char2idx.get(ch, unk) for ch in word]
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def _run(self, word: str, pos: str) -> str:
        src = self._encode_word(word)

        pos_id = torch.tensor(
            [self._pos2idx.get(pos, 0)],
            dtype=torch.long,
            device=self._device
        )

        tokens = beam_search(self._model, src, pos_id)
        return ids_to_str(tokens, self._idx2char)

    @torch.no_grad()
    def annotate(self, tokens: list[MorphToken]) -> list[MorphToken]:
        self._model.eval()

        result = []
        for token in tokens:
            lemma = self._run(token.text, token.pos or "NOUN")

            result.append(MorphToken(
                text=token.text,
                pos=token.pos,
                lemma=lemma,
                morph=token.morph
            ))

        return result
    
    def __call__(self, tokens: list[MorphToken]) -> list[MorphToken]:
        return self.annotate(tokens)
    

    def lemmatize(self, word: str, pos: str) -> str:
        """Lemmatize a single word given its POS tag."""
        return self._run(word, pos)