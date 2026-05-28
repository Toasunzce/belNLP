from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

MODELS_DIR = Path(__file__).resolve().parents[3] / "src" / "models"


class ModelRegistry:
    """
    Singleton — loads all heavy models once at application startup.
    Provides a single access point for all ML components.
    Implements the Singleton and Facade patterns.
    """

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return

        print("[ModelRegistry] loading models...")

        from gensim.models import FastText
        from belNLP.embeddings.static import GensimAdapter

        model = FastText.load(str(MODELS_DIR / "bel_ft.model"))
        self.fasttext = GensimAdapter(model.wv)

        print("[ModelRegistry] FastText OK")

        from belNLP.morphology.pos_tagger import POSTagger
        self.pos_tagger = POSTagger.load(MODELS_DIR / "POSTagger.pt")
        print("[ModelRegistry] POSTagger OK")

        from belNLP.morphology.lemmatizer import Lemmatizer
        self.lemmatizer = Lemmatizer.load(MODELS_DIR / "Lemmatizer.pt")
        print("[ModelRegistry] Lemmatizer OK")

        self._initialized = True
        print("[ModelRegistry] all models ready")

    @classmethod
    def get(cls) -> "ModelRegistry":
        return cls()


registry = ModelRegistry()