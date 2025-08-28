"""
Model configuration package for Portal Legends move classification.

This package contains configuration classes for different models used in the project.
Each model has its own configuration file that inherits from BaseModelConfig.
"""

from .base_config import BaseModelConfig
from .bert_config import BertConfig
from .distilbert_config import DistilBertConfig
from .roberta_config import RobertaConfig

__all__ = [
    'BaseModelConfig',
    'BertConfig',
    'DistilBertConfig',
    'RobertaConfig'
]
