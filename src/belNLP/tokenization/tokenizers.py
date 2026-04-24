import re
from collections import defaultdict

from tokenization.base import BaseTokenizer


"""
patterns:


"""


"""
Low-level module for text splitting (tokenization tasks).
Currently module supports the following tasks:

- RegexTokenizer:
  Tokenizer based on custom regular expressions. Uses a single pattern to extract tokens.

- WhitespaceTokenizer:
  Splits text by whitespace.
  Example:
    "We’re just grabbing coffee." -> ["We’re", "just", "grabbing", "coffee."]

- SentenceTokenizer:
  Splits text into sentences using simple punctuation rules.
  Note: does not handle edge cases like abbreviations (e.g. "Dr.", "e.g.").

- WordTokenizer:
  More advanced tokenizer for common separation cases:
    - words
    - numbers (keeps formats like ["3.1415", "1.000.000,00"])
    - punctuation
    - NLP special tokens (<SOS>, <PAD>, etc)

- BPETokenizer:
  <WORK IN PROGRESS>
"""



class RegexTokenizer(BaseTokenizer):
    """
    Base tokenizer using a single regular expression.

    Args:
        pattern (str): Regular expression used for token extraction

    Notes:
        - Pattern is compiled once during initialization
        - For complex tokenization, prefer using named groups (see WordTokenizer)
    """
    def __init__(self, pattern: str):
        self._pattern = pattern
        self._compiled_regex = re.compile(pattern)

    def tokenize(self, text: str) -> list[str]:
        return self._compiled_regex.findall(text)



class WhitespaceTokenizer(RegexTokenizer):
    """
    Tokenizer that splits text on whitespace.

    Used pattern: r"\\S+" (sequences of non-whitespace characters)

    Example:
        "Hello, world!" -> ["Hello,", "world!"]
    """
    def __init__(self):
        super().__init__(pattern=r"\S+")



class SentenceTokenizer(RegexTokenizer):
    """
    Naive sentence tokenizer based on punctuation. Does not handle abbreviations (e.g. "Dr.", "Mr.", "e.g.")

    Used pattern: r"[^\s][^.!?]+[.!?]" (sequences, separated by punctuation)

    Example:
        "Hello world. How are you?" -> ["Hello world.", "How are you?"]
    """
    def __init__(self):
        super().__init__(pattern=r"[^.!?]+[.!?]|[^.!?]+$")



class WordTokenizer(BaseTokenizer):
    """
    Rule-based tokenizer with support for words, numbers, punctuation and special tokens.

    Token types:
        - words
        - numbers (keeps formats like ["3.1415", "1.000.000,00"])
        - punctuation
        - NLP special tokens (<SOS>, <PAD>, etc)

    Example:
        "Hmmm, price is 1,234.56!" ->
        ["Hmmm", ",", "price", "is", "1,234.56", "!"]
    """
    def __init__(self):
        self._patterns = {
            "NUM":       r"\d+(?:[.,]\d+)+|\d+",
            "SPECIAL":   r"<[^>\s]+>",
            "WORD":      r"\w+(?:'\w+)*",
            "PUNCT":     r"[^\w\s]",
        }
        self._compiled_regex = re.compile(
            "|".join(f"(?P<{k}>{v})" for k, v in self._patterns.items()),
            re.UNICODE
        )

    def tokenize(self, text: str) -> list[str]:
        return [m.group() for m in self._compiled_regex.finditer(text)]



class BPETokenizer(BaseTokenizer):
    """
    
    """
    def __init__(self, left_spec='<', right_spec='>'):
        self._vocabulary: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self._left_spec = left_spec
        self._right_spec = right_spec
        self._cache: dict[str, tuple[str, ...]] = {}


    def fit(self, corpus: list[list[str]], vocab_size: int = 1024):
        self._cache = {}
        words = []
        charset = set()

        for sentence in corpus:
            for word in sentence:
                chars = list(word)
                charset.update(chars)
                words.append([self._left_spec] + chars + [self._right_spec])

        initial_vocab_size = len(charset) + 2
        if vocab_size <= initial_vocab_size:
            raise ValueError(
                f"vocab_size must be greater than initial character set ({initial_vocab_size})"
            )

        merges = []

        for _ in range(vocab_size - initial_vocab_size):
            pair_freq = defaultdict(int)

            for word in words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freq[pair] += 1

            if not pair_freq:
                break

            best_pair = max(pair_freq, key=pair_freq.get)   # FIXME pair.freq key error  # ty:ignore[no-matching-overload]
            merges.append(best_pair)

            new_words = []
            for word in words:
                i = 0
                new_word = []

                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                new_words.append(new_word)

            words = new_words

        self._merges = merges
        self._merge_ranks = {merge: i for i, merge in enumerate(self._merges)}

        vocab = set()
        for word in words:
            vocab.update(word)

        self._vocabulary = {
            token: idx for idx, token in enumerate(sorted(vocab))
        }


    def _tokenize(self, text: str) -> list[str]:
        words = text.split()
        tokens = []

        for word in words:
            encoded = self._encode_word(word)
            tokens.extend(encoded)

        return tokens


    def _encode_word(self, word: str) -> tuple[str, ...]:
        if word in self._cache:
            return self._cache[word]

        tokens = [self._left_spec] + list(word) + [self._right_spec]

        while True:
            best_rank = float('inf')
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_ranks.get(pair, float('inf'))

                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break

            tokens = (
                tokens[:best_idx]
                + [tokens[best_idx] + tokens[best_idx + 1]]
                + tokens[best_idx + 2:]
            )

        result = tuple(tokens)
        self._cache[word] = result
        return result