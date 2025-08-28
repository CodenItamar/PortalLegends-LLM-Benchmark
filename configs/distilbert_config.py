"""
DistilBERT model configuration for Portal Legends move classification.

This configuration file sets up the DistilBERT model with specific parameters and settings.
It inherits from the BaseModelConfig class and provides DistilBERT-specific configurations.

To use this configuration:
1. Import the DistilBertConfig class
2. Create an instance: config = DistilBertConfig()
3. Use the config object to initialize model and tokenizer

Example:
    from configs.distilbert_config import DistilBertConfig
    
    config = DistilBertConfig()
    model = config.model_class.from_pretrained(**config.get_model_params())
    tokenizer = config.tokenizer_class.from_pretrained(**config.get_tokenizer_params())
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from configs.base_config import BaseModelConfig

class DistilBertConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        
        # Model identifiers
        self.model_name = "distilbert-base-uncased"
        self.model_class = DistilBertForSequenceClassification
        self.tokenizer_class = DistilBertTokenizer
        
        # DistilBERT-specific parameters
        self.dropout = 0.1
        
        # Adjust learning rates for DistilBERT (typically needs slightly higher learning rates)
        self.learning_rates = [2e-5, 4e-5, 6e-5]
    
    def get_model_params(self):
        """
        Returns DistilBERT-specific model initialization parameters.
        Extends the base config with DistilBERT-specific parameters.
        """
        params = super().get_model_params()
        params.update({
            'dropout': self.dropout
        })
        return params
