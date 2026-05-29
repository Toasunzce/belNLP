"""
belNLP.orchestration
~~~~~~~~~~~~~~~~~~~~
Модуль оркестрации — собирает компоненты библиотеки в сквозные пайплайны.

Public API:
    Annotation      — сквозной объект данных
    BasePipelineStep — базовый шаг пайплайна
    PreprocessStep  — препроцессинг текста
    TokenizeStep    — токенизация
    FilterStep      — фильтрация tokens
    FilterMorphStep — фильтрация morph_tokens + синхронизация tokens
    POSTagStep      — POS-тэггинг
    LemmatizeStep   — лемматизация
    EmbedStep       — эмбеддинг
    Pipeline        — исполнитель шагов
    PipelineBuilder — fluent-конструктор пайплайнов
    PipelineFactory — фабрика предустановленных пайплайнов
"""

from belNLP.orchestration.annotation import Annotation
from belNLP.orchestration.steps import (
    BasePipelineStep,
    PreprocessStep,
    TokenizeStep,
    FilterStep,
    FilterMorphStep,
    POSTagStep,
    LemmatizeStep,
    EmbedStep,
)
from belNLP.orchestration.pipeline import Pipeline, PipelineBuilder
from belNLP.orchestration.factory import PipelineFactory

__all__ = [
    "Annotation",
    "BasePipelineStep",
    "PreprocessStep",
    "TokenizeStep",
    "FilterStep",
    "FilterMorphStep",
    "POSTagStep",
    "LemmatizeStep",
    "EmbedStep",
    "Pipeline",
    "PipelineBuilder",
    "PipelineFactory",
]
