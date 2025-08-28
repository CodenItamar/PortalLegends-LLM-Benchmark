"""
Model Registry for Portal Legends move classification project.

This module provides a centralized registry for all model configurations.
It handles the registration and retrieval of different model configurations.

To add a new model:
1. Create a new configuration file in the configs directory
2. Import the configuration class here
3. Register the model in the DEFAULT_MODELS dictionary

Example:
    from configs.new_model_config import NewModelConfig
    
    registry = ModelRegistry()
    registry.register_model('new_model', NewModelConfig())
    model_config = registry.get_model('new_model')
"""

from typing import Dict, Type
from configs.base_config import BaseModelConfig
from configs.bert_config import BertConfig
from configs.distilbert_config import DistilBertConfig
from configs.roberta_config import RobertaConfig

class ModelRegistry:
    def __init__(self):
        """Initialize an empty model registry."""
        self.configs: Dict[str, BaseModelConfig] = {}
        self._register_default_models()
    
    def register_model(self, name: str, config: BaseModelConfig) -> None:
        """
        Register a new model configuration.
        
        Args:
            name: Unique identifier for the model
            config: Configuration instance inheriting from BaseModelConfig
        
        Raises:
            ValueError: If a model with the given name is already registered
            TypeError: If the config is not an instance of BaseModelConfig
        """
        if name in self.configs:
            raise ValueError(f"Model '{name}' is already registered")
        
        if not isinstance(config, BaseModelConfig):
            raise TypeError("Config must be an instance of BaseModelConfig")
        
        self.configs[name] = config
    
    def get_model(self, name: str) -> BaseModelConfig:
        """
        Retrieve a model configuration by name.
        
        Args:
            name: The identifier of the model to retrieve
            
        Returns:
            The model configuration instance
            
        Raises:
            KeyError: If no model is registered with the given name
        """
        if name not in self.configs:
            raise KeyError(f"No model registered with name '{name}'")
        
        return self.configs[name]
    
    def list_models(self) -> list:
        """Return a list of all registered model names."""
        return list(self.configs.keys())
    
    def _register_default_models(self) -> None:
        """Register the default set of models."""
        DEFAULT_MODELS = {
            'bert': BertConfig(),
            'distilbert': DistilBertConfig(),
            'roberta': RobertaConfig()
        }
        
        for name, config in DEFAULT_MODELS.items():
            try:
                self.register_model(name, config)
            except (ValueError, TypeError) as e:
                print(f"Failed to register {name}: {str(e)}")
