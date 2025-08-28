"""
RoBERTa model configuration for Portal Legends move classification.

This configuration file sets up the RoBERTa model with specific parameters and settings.
It inherits from the BaseModelConfig class and provides RoBERTa-specific configurations.

To use this configuration:
1. Import the RobertaConfig class
2. Create an instance: config = RobertaConfig()
3. Use the config object to initialize model and tokenizer

Example:
    from configs.roberta_config import RobertaConfig
    
    config = RobertaConfig()
    model = config.model_class.from_pretrained(**config.get_model_params())
    tokenizer = config.tokenizer_class.from_pretrained(**config.get_tokenizer_params())
"""

from transformers import RobertaForSequenceClassification, RobertaTokenizer
from configs.base_config import BaseModelConfig

class RobertaConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        
        # Model identifiers
        self.model_name = "roberta-base"
        self.model_class = RobertaForSequenceClassification
        self.tokenizer_class = RobertaTokenizer
        
        # RoBERTa-specific parameters
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        
        # RoBERTa often works better with slightly different learning rates
        self.learning_rates = [1e-5, 2e-5, 3e-5]
        
    def get_model_params(self):
        """
        Returns RoBERTa-specific model initialization parameters.
        Extends the base config with RoBERTa-specific parameters.
        """
        params = super().get_model_params()
        params.update({
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob
        })
        return params
        
    def get_tokenizer_params(self):
        """
        Returns RoBERTa-specific tokenizer initialization parameters.
        RoBERTa uses a different tokenization strategy than BERT/DistilBERT.
        """
        params = super().get_tokenizer_params()
        params.update({
            'add_prefix_space': True  # RoBERTa-specific parameter
        })
        return params
