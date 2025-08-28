"""
Base configuration class for all models in the Portal Legends move classification project.
This serves as a template and defines common parameters and interfaces that all model configs must implement.

To create a new model configuration:
1. Create a new file in the configs directory (e.g., new_model_config.py)
2. Import this base class
3. Create your model class inheriting from BaseModelConfig
4. Override necessary parameters and methods

Example:
    class NewModelConfig(BaseModelConfig):
        def __init__(self):
            super().__init__()
            self.model_name = "new-model-name"
            self.model_class = NewModelClass
            self.tokenizer_class = NewTokenizerClass
"""

class BaseModelConfig:
    def __init__(self):
        # Model identifiers
        self.model_name = None  # Name of the model (e.g., "bert-base-uncased")
        self.model_class = None  # The actual model class from transformers
        self.tokenizer_class = None  # The tokenizer class from transformers
        
        # Training hyperparameters
        self.learning_rates = [1e-5, 3e-5, 5e-5]  # Learning rates to try during hyperparameter search
        self.batch_sizes = [16, 32]  # Batch sizes to try during hyperparameter search
        self.max_epochs = 10  # Maximum number of training epochs
        self.early_stopping_patience = 3  # Number of epochs to wait before early stopping
        self.max_length = 512  # Maximum sequence length for tokenization
        
        # Optimizer parameters
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        
        # Training settings
        self.num_labels = 2  # Binary classification (Legal/Illegal)
        self.device = 'cuda'  # Will be updated based on availability
        
    def get_model_params(self):
        """
        Returns the model initialization parameters.
        Override this method if your model needs specific parameters.
        """
        return {
            'num_labels': self.num_labels,
            'pretrained_model_name_or_path': self.model_name
        }
    
    def get_tokenizer_params(self):
        """
        Returns the tokenizer initialization parameters.
        Override this method if your tokenizer needs specific parameters.
        """
        return {
            'pretrained_model_name_or_path': self.model_name,
            'max_length': self.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_tensors': 'pt'
        }
