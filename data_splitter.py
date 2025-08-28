#!/usr/bin/env python3
"""
Data splitter for Portal Legends move classification project.

This script splits the dataset into persistent train/validation/test CSV files
that can be reused across multiple training and inference runs.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess_dataset(path: str):
    """
    Load and preprocess the Portal Legends dataset.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        Processed dataset with labels
    """
    df = pd.read_csv(path)
    
    # Convert labels to numeric
    df['label'] = (df['validity'] == 'Legal').astype(int)
    
    return df

def split_and_save_dataset(input_path: str, output_dir: str = 'data'):
    """
    Split dataset into train/validation/test sets and save as persistent CSV files.
    
    Args:
        input_path: Path to the input dataset
        output_dir: Directory to save the split datasets
    """
    print("Loading and preprocessing dataset...")
    df = load_and_preprocess_dataset(input_path)
    print(f"Loaded {len(df)} examples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    print(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Save to persistent CSV files with constant names
    train_path = os.path.join(output_dir, 'train_data.csv')
    val_path = os.path.join(output_dir, 'val_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved datasets:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")
    
    # Print class distribution for each split
    print("\nClass distribution:")
    print(f"Train - Legal: {train_df['label'].sum()}, Illegal: {len(train_df) - train_df['label'].sum()}")
    print(f"Val - Legal: {val_df['label'].sum()}, Illegal: {len(val_df) - val_df['label'].sum()}")
    print(f"Test - Legal: {test_df['label'].sum()}, Illegal: {len(test_df) - test_df['label'].sum()}")
    
    return train_path, val_path, test_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split Portal Legends dataset into persistent train/val/test files")
    parser.add_argument(
        "--input", 
        default="PortalLegendsMovesTagged1.csv",
        help="Input dataset file path"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for split datasets"
    )
    
    args = parser.parse_args()
    
    split_and_save_dataset(args.input, args.output_dir)
    print("\nData splitting completed successfully!")
