"""
BERT model configuration for Portal Legends move classification.

This configuration file sets up the BERT model with specific parameters and settings.
It inherits from the BaseModelConfig class and provides BERT-specific configurations.

To use this configuration:
1. Import the BertConfig class
2. Create an instance: config = BertConfig()
3. Use the config object to initialize model and tokenizer

Example:
    from configs.bert_config import BertConfig
    
    config = BertConfig()
    model = config.model_class.from_pretrained(**config.get_model_params())
    tokenizer = config.tokenizer_class.from_pretrained(**config.get_tokenizer_params())
"""

from transformers import BertForSequenceClassification, BertTokenizer
from configs.base_config import BaseModelConfig

class BertConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        
        # Model identifiers
        self.model_name = "bert-base-uncased"
        self.model_class = BertForSequenceClassification
        self.tokenizer_class = BertTokenizer
        
        # BERT-specific parameters
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        
    def get_model_params(self):
        """
        Returns BERT-specific model initialization parameters.
        Extends the base config with BERT-specific parameters.
        """
        params = super().get_model_params()
        params.update({
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob
        })
        return params
