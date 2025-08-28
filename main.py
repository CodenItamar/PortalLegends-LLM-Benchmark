"""
Main entry point for Portal Legends move classification project.

This script demonstrates how to:
1. Load and preprocess the Portal Legends dataset
2. Train and evaluate multiple models
3. Perform hyperparameter search
4. Compare model performances

To run experiments with specific models:
    python main.py --models bert distilbert roberta
"""

import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model_registry import ModelRegistry

def load_dataset(path: str):
    """
    Load and preprocess the Portal Legends dataset.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        Processed dataset with combined prompts and labels
    """
    df = pd.read_csv(path)
    
    # Combine board states and move into a single prompt
    # df['prompt'] = (
    #     df['PreMoveBoardState'] + " " +
    #     df['Move'] + " " +
    #     df['PostMoveBoardState']
    # )
    
    # Convert labels to numeric
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

def train_and_evaluate(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train a model, perform hyperparameter search, and evaluate on test set.
    
    Args:
        model_name: Name of the model to use
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        
    Returns:
        Test accuracy
    """
    from src.trainer import Trainer, create_dataloaders
    from datetime import datetime
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get model configuration
    registry = ModelRegistry()
    config = registry.get_model(model_name)
    
    # Initialize model and tokenizer
    model = config.model_class.from_pretrained(**config.get_model_params())
    tokenizer = config.tokenizer_class.from_pretrained(**config.get_tokenizer_params())
    
    # Prepare data
    train_texts = train_df['prompt'].tolist()
    val_texts = val_df['prompt'].tolist()
    test_texts = test_df['prompt'].tolist()
    
    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        tokenizer,
        batch_size=config.batch_sizes[0]  # Start with first batch size
    )
    
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Train and perform hyperparameter search
    best_val_acc = 0
    best_lr = None
    
    print(f"\nPerforming hyperparameter search for {model_name}...")
    for lr in config.learning_rates:
        print(f"\nTrying learning rate: {lr}")
        val_acc = trainer.train(train_dataloader, val_dataloader, learning_rate=lr)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
    
    print(f"\nBest learning rate: {best_lr}")
    
    # Evaluate on test set and save predictions
    test_accuracy = trainer.evaluate_and_save_predictions(test_dataloader, test_df, model_name, timestamp)
    
    return test_accuracy

def main(models_to_run: list):
    """
    Main execution function.
    
    Args:
        models_to_run: List of model names to experiment with
    """
    print("Starting Portal Legends move classification experiments...")
    
    # Load dataset
    print("Loading dataset...")
    df = load_dataset("PortalLegendsMovesTagged2.csv")
    
    # Split dataset
    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)
    print(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Train and evaluate each model
    results = {}
    for model_name in models_to_run:
        print(f"\nExperimenting with {model_name}...")
        accuracy = train_and_evaluate(model_name, train_df, val_df, test_df)
        results[model_name] = accuracy
    
    # Print results
    print("\nFinal Results:")
    print("-" * 40)
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f} accuracy")
    
    print(f"\nAll results have been saved to the 'results' directory with timestamp information.")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Portal Legends Move Classification")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert", "distilbert", "roberta"],
        help="List of models to run experiments with"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    main(args.models)
