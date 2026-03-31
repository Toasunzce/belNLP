import re
from base import BaseTokenizer



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
        super().__init__(pattern=r"[^\s][^.!?]+[.!?]")



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



# class BPETokenizer(BaseTokenizer):
#     """
    
#     """
#     def __init__(self, ...):
#         ...

#     def train(self, ...):
#         ...

#     def merge(self, ...):
#         ...

#     def tokenize(self, text: str) -> list[str]:
#         ...

    