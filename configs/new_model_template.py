"""
Template for adding new models to the Portal Legends move classification project.

To add a new model:
1. Copy this template file and rename it (e.g., my_model_config.py)
2. Replace NewModelConfig with your model's name
3. Update the model-specific parameters and imports
4. Import and register your model in src/model_registry.py

Required Steps:
1. Import your model and tokenizer classes from transformers
2. Update model identifiers (model_name, model_class, tokenizer_class)
3. Add any model-specific parameters
4. Override get_model_params() if your model needs special parameters
5. Override get_tokenizer_params() if your tokenizer needs special parameters

Example implementation below:
"""

from transformers import PreTrainedModel, PreTrainedTokenizer  # Replace with your model's actual classes
from configs.base_config import BaseModelConfig

class NewModelConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        
        # Model identifiers - REQUIRED
        self.model_name = "new-model-name"  # The model name from HuggingFace hub
        self.model_class = PreTrainedModel  # Your model class
        self.tokenizer_class = PreTrainedTokenizer  # Your tokenizer class
        
        # Model-specific parameters - OPTIONAL
        # Add any parameters specific to your model
        self.special_param1 = 0.1
        self.special_param2 = 0.2
        
        # Hyperparameter adjustments - OPTIONAL
        # Adjust if your model requires different hyperparameters
        self.learning_rates = [1e-5, 3e-5, 5e-5]
        self.batch_sizes = [16, 32]
        self.max_length = 512
    
    def get_model_params(self):
        """
        Returns model-specific initialization parameters.
        Override this if your model needs special parameters.
        """
        params = super().get_model_params()
        params.update({
            'special_param1': self.special_param1,
            'special_param2': self.special_param2
            # Add any other model-specific parameters
        })
        return params
    
    def get_tokenizer_params(self):
        """
        Returns tokenizer-specific initialization parameters.
        Override this if your tokenizer needs special parameters.
        """
        params = super().get_tokenizer_params()
        params.update({
            # Add any tokenizer-specific parameters
        })
        return params

"""
To register your new model in the registry:

1. Add import in src/model_registry.py:
   from configs.my_model_config import MyModelConfig

2. Add to DEFAULT_MODELS in ModelRegistry._register_default_models():
   DEFAULT_MODELS = {
       ...,
       'my_model': MyModelConfig()
   }

3. Use your model:
   registry = ModelRegistry()
   my_model_config = registry.get_model('my_model')
"""
