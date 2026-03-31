from tokenizers import WhitespaceTokenizer


tokenizer = WhitespaceTokenizer()


text = """We’re just grabbing coffee."""

print(tokenizer.tokenize(text))