import re
from collections import defaultdict

from belNLP.tokenization.base import BaseTokenizer


class RegexTokenizer(BaseTokenizer):
    """Tokenizes text using a single compiled regular expression.

    Example:
        >>> RegexTokenizer(r"\\w+").tokenize("прывет, свет!") == ["прывет", "свет"]
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._compiled_regex = re.compile(pattern)

    def tokenize(self, text: str) -> list[str]:
        return self._compiled_regex.findall(text)


class WhitespaceTokenizer(RegexTokenizer):
    """Splits text on whitespace. Keeps punctuation attached to words.

    Example:
        >>> WhitespaceTokenizer().tokenize("Прывет, свет!") == ["Прывет,", "свет!"]
    """

    def __init__(self) -> None:
        super().__init__(pattern=r"\S+")


class SentenceTokenizer(RegexTokenizer):
    """Splits text into sentences on .!? — does not handle abbreviations.

    Example:
        >>> SentenceTokenizer().tokenize("Прывет. Як справы?") == ["Прывет.", "Як справы?"]
    """

    def __init__(self) -> None:
        super().__init__(pattern=r"[^.!?]+[.!?]|[^.!?]+$")


class WordTokenizer(BaseTokenizer):
    """Splits text into words, numbers, punctuation and NLP special tokens.

    Token types: WORD, NUM (e.g. 1,234.56), PUNCT, SPECIAL (<PAD>, <UNK>).

    Example:
        >>> WordTokenizer().tokenize("Цана: 1,234.56!") == ["Цана", ":", "1,234.56", "!"]
    """

    def __init__(self) -> None:
        self._patterns = {
            "NUM":     r"\d+(?:[.,]\d+)+|\d+",
            "SPECIAL": r"<[^>\s]+>",
            "WORD":    r"\w+(?:'\w+)*",
            "PUNCT":   r"[^\w\s]",
        }
        self._compiled_regex = re.compile(
            "|".join(f"(?P<{k}>{v})" for k, v in self._patterns.items()),
            re.UNICODE,
        )

    def _tokenize(self, text: str) -> list[str]:
        return [m.group() for m in self._compiled_regex.finditer(text)]


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding tokenizer. Must be trained with fit() before use.

    Example:
        >>> bpe = BPETokenizer()
        >>> bpe.fit([["прывет", "свет"], ["кот", "бяжыць"]], vocab_size=200)
        >>> bpe.tokenize("прывет свет")
    """

    def __init__(self, left_spec: str = "<", right_spec: str = ">") -> None:
        self._vocabulary: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self._left_spec = left_spec
        self._right_spec = right_spec
        self._cache: dict[str, tuple[str, ...]] = {}

    def fit(self, corpus: list[list[str]], vocab_size: int = 1024) -> None:
        """Train BPE merges on a tokenized corpus."""
        self._cache = {}
        words: list[list[str]] = []
        charset: set[str] = set()

        for sentence in corpus:
            for word in sentence:
                chars = list(word)
                charset.update(chars)
                words.append([self._left_spec] + chars + [self._right_spec])

        initial_vocab_size = len(charset) + 2
        if vocab_size <= initial_vocab_size:
            raise ValueError(
                f"vocab_size must be > initial character set size ({initial_vocab_size})"
            )

        merges: list[tuple[str, str]] = []

        for _ in range(vocab_size - initial_vocab_size):
            pair_freq: dict[tuple[str, str], int] = defaultdict(int)
            for word in words:
                for i in range(len(word) - 1):
                    pair_freq[(word[i], word[i + 1])] += 1

            if not pair_freq:
                break

            best_pair = max(pair_freq, key=lambda p: pair_freq[p])
            merges.append(best_pair)

            new_words = []
            for word in words:
                i, new_word = 0, []
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
        self._merge_ranks = {merge: i for i, merge in enumerate(merges)}

        vocab: set[str] = set()
        for word in words:
            vocab.update(word)
        self._vocabulary = {token: idx for idx, token in enumerate(sorted(vocab))}

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self._encode_word(word))
        return tokens

    def _encode_word(self, word: str) -> tuple[str, ...]:
        if word in self._cache:
            return self._cache[word]

        tokens: list[str] = [self._left_spec] + list(word) + [self._right_spec]

        while True:
            best_rank, best_idx = float("inf"), -1
            for i in range(len(tokens) - 1):
                rank = self._merge_ranks.get((tokens[i], tokens[i + 1]), float("inf"))
                if rank < best_rank:
                    best_rank, best_idx = rank, i
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
