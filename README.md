# Portal Legends Move Classification

This project investigates whether Large Language Models (LLMs) can infer rule-based reasoning from text alone, using Portal Legends game moves as a test case. The system classifies moves as legal or illegal based on natural language descriptions of game states.

## Project Structure

```
ANLP-Project-Pipeline/
├── configs/                    # Model configurations
│   ├── base_config.py         # Base configuration class
│   ├── bert_config.py         # BERT-specific configuration
│   ├── distilbert_config.py   # DistilBERT-specific configuration
│   ├── roberta_config.py      # RoBERTa-specific configuration
│   └── new_model_template.py  # Template for adding new models
├── src/                       # Source code
│   ├── model_registry.py      # Model configuration registry
│   └── [future modules]       # Training, evaluation, etc.
├── main.py                    # Main execution script
└── README.md                  # Project documentation
```

## Setup

1. **Dependencies**
   ```bash
   pip install torch transformers pandas scikit-learn
   ```

2. **Dataset**
   - Place `PortalLegendsMovesTagged.csv` in the project root directory
   - Dataset should contain columns:
     - PreMoveBoardState
     - Move
     - PostMoveBoardState
     - VALIDITY (Legal/Illegal)

## Usage

Run experiments with default models:
```bash
python main.py
```

Specify specific models to run:
```bash
python main.py --models bert distilbert
```

## Adding New Models

1. **Create Configuration File**
   - Copy `configs/new_model_template.py` to `configs/your_model_config.py`
   - Follow the template instructions to implement your model configuration

2. **Register the Model**
   - Add your model import to `src/model_registry.py`
   - Add your model to `DEFAULT_MODELS` in the registry

Example:
```python
# configs/my_model_config.py
from transformers import MyModel, MyTokenizer
from configs.base_config import BaseModelConfig

class MyModelConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "my-model-name"
        self.model_class = MyModel
        self.tokenizer_class = MyTokenizer
```

## Model Configurations

Each model configuration inherits from `BaseModelConfig` and can specify:
- Model and tokenizer classes
- Model-specific parameters
- Custom hyperparameter ranges
- Special tokenization requirements

### Current Models

1. **BERT**
   - Base model: `bert-base-uncased`
   - Learning rates: [1e-5, 3e-5, 5e-5]

2. **DistilBERT**
   - Base model: `distilbert-base-uncased`
   - Learning rates: [2e-5, 4e-5, 6e-5]

3. **RoBERTa**
   - Base model: `roberta-base`
   - Learning rates: [1e-5, 2e-5, 3e-5]
   - Special tokenization: `add_prefix_space=True`

## Training Process

The system:
1. Loads and preprocesses the dataset
2. Splits data into train (60%), validation (20%), and test (20%) sets
3. For each model:
   - Trains on training data
   - Performs hyperparameter search using validation data
   - Evaluates final performance on test data
4. Compares model performances

## Future Enhancements

- Training and evaluation pipelines
- Model performance visualization
- Additional models (T5, LLaMA 7B)
- Few-shot learning experiments
- Qualitative error analysis tools
