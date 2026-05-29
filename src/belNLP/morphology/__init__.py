"""
belNLP.morphology
~~~~~~~~~~~~~~~~~
POS tagging and lemmatization for Belarusian text.

Example usage:
    from belNLP.morphology.pos_tagger import POSTagger
    from belNLP.morphology.lemmatizer import Lemmatizer
    from belNLP.morphology.base import MorphToken

    tagger = POSTagger.load("models/POSTagger.pt")
    lemmatizer = Lemmatizer.load("models/Lemmatizer.pt")

    tokens = ["я", "іду", "дадому"]
    morph = tagger.annotate(tokens)        # -> [MorphToken(pos=...), ...]
    morph = lemmatizer.annotate(morph)     # -> lemma filled in each token

    morph[1].pos    # -> "VERB"
    morph[1].lemma  # -> "ісці"
"""

from belNLP.morphology.base import MorphToken, BaseAnnotator
from belNLP.morphology.pos_tagger import POSTagger
from belNLP.morphology.lemmatizer import Lemmatizer

__all__ = [
    "MorphToken",
    "BaseAnnotator",
    "POSTagger",
    "Lemmatizer",
]
