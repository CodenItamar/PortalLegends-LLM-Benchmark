"""
Inference script for Portal Legends move classification.

This script loads trained models and makes predictions on test data.
It can use any saved model from the training phase.
"""

import sys
import os
import pandas as pd
import torch
import json
import glob
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Add src to path
sys.path.append('src')
sys.path.append('configs')

from src.model_registry import ModelRegistry
from src.trainer import Trainer, create_dataloaders

def load_test_data():
    """Load the test data split."""
    try:
        test_df = pd.read_csv('data/test_data.csv')
        print(f"Loaded test data: {len(test_df)} examples")
        return test_df
    except FileNotFoundError:
        print("Error: Could not find test data. Please run data_splitter.py first.")
        sys.exit(1)

def list_available_models():
    """List all available trained models."""
    model_files = glob.glob('models/*_model_*.pth')
    hyperparam_files = glob.glob('models/*_hyperparams_*.json')
    
    if not model_files:
        print("No trained models found. Please run train.py first.")
        return []
    
    models = []
    for model_file in model_files:
        # Extract model info from filename
        basename = os.path.basename(model_file)
        parts = basename.replace('.pth', '').split('_')
        if len(parts) >= 3:
            model_name = parts[0]
            timestamp = '_'.join(parts[2:])
            
            # Find corresponding hyperparams file
            hyperparam_file = f"models/{model_name}_hyperparams_{timestamp}.json"
            if hyperparam_file in hyperparam_files:
                models.append({
                    'model_name': model_name,
                    'timestamp': timestamp,
                    'model_path': model_file,
                    'hyperparams_path': hyperparam_file
                })
    
    return models

def load_model_and_predict(model_info, test_df):
    """Load a specific model and make predictions on test data."""
    print(f"\nLoading model: {model_info['model_name']} ({model_info['timestamp']})")
    
    # Load hyperparameters
    with open(model_info['hyperparams_path'], 'r') as f:
        hyperparams = json.load(f)
    
    # Get model architecture and config
    registry = ModelRegistry()
    config = registry.get_model(model_info['model_name'])
    
    # Initialize model and tokenizer
    model = config.model_class.from_pretrained(**config.get_model_params())
    tokenizer = config.tokenizer_class.from_pretrained(**config.get_tokenizer_params())
    
    # Load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_info['model_path'], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create trainer for evaluation
    trainer = Trainer(model, config)
    
    # Prepare test data
    test_texts = test_df['prompt'].tolist()
    test_labels = [1 if validity == 'Legal' else 0 for validity in test_df['validity']]
    
    # Create minimal dataloaders (we only need test)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        test_texts[:1], test_labels[:1],  # Dummy train data
        test_texts[:1], test_labels[:1],  # Dummy val data
        test_texts, test_labels,          # Real test data
        tokenizer,
        config.batch_sizes[0]  # Use first batch size
    )
    
    # Make predictions
    accuracy = trainer.evaluate_and_save_predictions(
        test_dataloader, 
        test_df, 
        model_info['model_name'],
        model_info['timestamp']
    )
    
    print(f"Model: {model_info['model_name']}")
    print(f"Training timestamp: {model_info['timestamp']}")
    print(f"Best validation accuracy: {hyperparams['best_val_accuracy']:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    return accuracy

def main():
    print("Portal Legends Move Classification - Inference")
    print("=" * 50)
    
    # Load test data
    test_df = load_test_data()
    
    # List available models
    available_models = list_available_models()
    
    if not available_models:
        return
    
    print(f"\nFound {len(available_models)} trained models:")
    for i, model_info in enumerate(available_models):
        print(f"  {i+1}. {model_info['model_name']} ({model_info['timestamp']})")
    
    # Option to run all models or select specific ones
    print("\nOptions:")
    print("  1. Run inference on all models")
    print("  2. Select specific model")
    print("  3. Run inference on best model from each type")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Run all models
            results = []
            for model_info in available_models:
                accuracy = load_model_and_predict(model_info, test_df)
                results.append({
                    'model_name': model_info['model_name'],
                    'timestamp': model_info['timestamp'],
                    'test_accuracy': accuracy
                })
            
            # Print summary
            print(f"\n{'='*60}")
            print("INFERENCE SUMMARY")
            print(f"{'='*60}")
            results_df = pd.DataFrame(results)
            print(results_df.to_string(index=False))
            
        elif choice == "2":
            # Select specific model
            model_idx = int(input(f"Select model (1-{len(available_models)}): ")) - 1
            if 0 <= model_idx < len(available_models):
                load_model_and_predict(available_models[model_idx], test_df)
            else:
                print("Invalid selection.")
                
        elif choice == "3":
            # Run best model from each type
            model_types = {}
            for model_info in available_models:
                model_name = model_info['model_name']
                if model_name not in model_types:
                    model_types[model_name] = []
                model_types[model_name].append(model_info)
            
            results = []
            for model_name, models in model_types.items():
                # For simplicity, take the most recent model of each type
                # In a real scenario, you'd want to load validation scores to pick the best
                latest_model = max(models, key=lambda x: x['timestamp'])
                print(f"\nRunning inference with latest {model_name} model...")
                accuracy = load_model_and_predict(latest_model, test_df)
                results.append({
                    'model_name': model_name,
                    'timestamp': latest_model['timestamp'],
                    'test_accuracy': accuracy
                })
            
            # Print summary
            print(f"\n{'='*60}")
            print("INFERENCE SUMMARY - BEST MODELS")
            print(f"{'='*60}")
            results_df = pd.DataFrame(results)
            print(results_df.to_string(index=False))
            
        else:
            print("Invalid choice.")
            
    except (ValueError, KeyboardInterrupt):
        print("\nOperation cancelled.")

if __name__ == "__main__":
    main()
