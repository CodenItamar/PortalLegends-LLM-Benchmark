"""
Training script for Portal Legends move classification.

This script handles training and validation using persistent CSV data splits.
It saves the trained model weights and hyperparameters for later inference.
"""

import sys
import os
import pandas as pd
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')
sys.path.append('configs')

from src.model_registry import ModelRegistry
from src.trainer import Trainer, create_dataloaders

def load_dataset(path: str):
    """
    Load and preprocess the Portal Legends dataset.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        Processed dataset with combined prompts and labels
    """
    df = pd.read_csv(path)
    
    # Convert labels to numeric for stratification
    df['label'] = (df['validity'] == 'Legal').astype(int)
    
    return df

def split_dataset(df: pd.DataFrame):
    """
    Split dataset into train, validation, and test sets (60/20/20).
    
    Args:
        df: Input DataFrame
        
    Returns:
        train, val, test DataFrames
    """
    # First split: 60% train, 40% remaining
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.4, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Second split: Split remaining 40% into half (20% each for val and test)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['label']
    )
    
    return train_df, val_df, test_df

def create_or_load_data_splits():
    """Create persistent data splits or load existing ones."""
    data_dir = 'data'
    train_path = os.path.join(data_dir, 'train_data.csv')
    val_path = os.path.join(data_dir, 'val_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    
    # Check if all split files exist
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Loading existing data splits...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        print("Creating new data splits from original dataset...")
        
        # Load original dataset
        original_dataset = "PortalLegendsMovesTagged2.csv"
        if not os.path.exists(original_dataset):
            print(f"Error: Original dataset '{original_dataset}' not found!")
            sys.exit(1)
        
        df = load_dataset(original_dataset)
        print(f"Loaded dataset with {len(df)} examples")
        
        # Split the dataset
        train_df, val_df, test_df = split_dataset(df)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Save splits to CSV files (overwriting any existing files)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Created and saved new data splits:")
        print(f"  Train data saved to: {train_path}")
        print(f"  Validation data saved to: {val_path}")
        print(f"  Test data saved to: {test_path}")
    
    print(f"Data splits loaded:")
    print(f"  Training: {len(train_df)} examples")
    print(f"  Validation: {len(val_df)} examples")
    print(f"  Test: {len(test_df)} examples")
    
    return train_df, val_df, test_df

def prepare_data(df, tokenizer):
    """Prepare texts and labels from DataFrame."""
    texts = df['prompt'].tolist()
    # Convert validity to binary labels (Legal=1, Illegal=0)
    labels = [1 if validity == 'Legal' else 0 for validity in df['validity']]
    return texts, labels

def main():
    # Configuration
    model_names = ['bert', 'distilbert', 'roberta']
    learning_rates = [2e-5, 3e-5, 5e-5]
    num_epochs = 10
    
    # Load data splits
    train_df, val_df, test_df = create_or_load_data_splits()
    
    # Store results for comparison
    results = []
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} model")
        print(f"{'='*60}")
        
        # Get model and config
        registry = ModelRegistry()
        config = registry.get_model(model_name)
        
        # Initialize model and tokenizer
        model = config.model_class.from_pretrained(**config.get_model_params())
        tokenizer = config.tokenizer_class.from_pretrained(**config.get_tokenizer_params())
        
        # Prepare data
        train_texts, train_labels = prepare_data(train_df, tokenizer)
        val_texts, val_labels = prepare_data(val_df, tokenizer)
        test_texts, test_labels = prepare_data(test_df, tokenizer)
        
        best_lr = None
        best_val_acc = 0
        best_trainer = None
        
        # Hyperparameter search
        for lr in learning_rates:
            print(f"\nTesting learning rate: {lr}")
            
            # Create fresh model instance for each learning rate
            registry = ModelRegistry()
            config = registry.get_model(model_name)
            model = config.model_class.from_pretrained(**config.get_model_params())
            trainer = Trainer(model, config)
            
            # Create data loaders
            train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                tokenizer,
                config.batch_sizes[0]  # Use first batch size
            )
            
            # Train model
            val_acc = trainer.train(train_dataloader, val_dataloader, lr, num_epochs)
            
            print(f"Validation accuracy for lr={lr}: {val_acc:.4f}")
            
            # Keep track of best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_trainer = trainer
        
        print(f"\nBest configuration for {model_name}:")
        print(f"  Learning rate: {best_lr}")
        print(f"  Validation accuracy: {best_val_acc:.4f}")
        
        # Save the best model
        if best_trainer is not None:
            save_info = best_trainer.save_model(model_name, best_lr, best_val_acc)
            
            # Store results
            results.append({
                'model_name': model_name,
                'best_learning_rate': best_lr,
                'best_val_accuracy': best_val_acc,
                'model_path': save_info['model_path'],
                'hyperparams_path': save_info['hyperparams_path'],
                'timestamp': save_info['timestamp']
            })
    
    # Print summary of all models
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save training summary
    os.makedirs('results', exist_ok=True)
    summary_file = f"results/training_summary_{results[0]['timestamp']}.csv"
    results_df.to_csv(summary_file, index=False)
    print(f"\nTraining summary saved to {summary_file}")
    
    # Find best overall model
    best_model = results_df.loc[results_df['best_val_accuracy'].idxmax()]
    print(f"\nBest overall model: {best_model['model_name']} with accuracy {best_model['best_val_accuracy']:.4f}")

if __name__ == "__main__":
    main()
