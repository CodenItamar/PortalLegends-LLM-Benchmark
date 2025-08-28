"""
Trainer module for Portal Legends move classification.

This module contains the training, validation, and hyperparameter search logic
for the model training pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
from datetime import datetime

class PortalLegendsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            loss.backward()
            optimizer.step()
        
        return total_loss / len(dataloader), correct / total
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def evaluate_and_save_predictions(self, dataloader, test_df, model_name, timestamp=None):
        """
        Perform evaluation and save predictions to CSV.
        
        Args:
            dataloader: Test dataloader
            test_df: Original test DataFrame with all columns
            model_name: Name of the model being evaluated
            timestamp: Optional timestamp string
            
        Returns:
            Test accuracy
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get probabilities using softmax
                probabilities = torch.softmax(outputs.logits, dim=1)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Create results DataFrame
        results_df = test_df.copy()
        results_df['model_name'] = model_name
        results_df['predicted_label'] = all_predictions
        results_df['confidence_illegal'] = [prob[0] for prob in all_probabilities]
        results_df['confidence_legal'] = [prob[1] for prob in all_probabilities]
        results_df['predicted_validity'] = [('Legal' if pred == 1 else 'Illegal') for pred in all_predictions]
        
        # Calculate accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Save to CSV
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs('results', exist_ok=True)
        filename = f"results/test_predictions_{model_name}_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"Test predictions saved to {filename}")
        print(f"Test accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def train(self, train_dataloader, val_dataloader, learning_rate, num_epochs=10):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        best_val_acc = 0
        best_model = None
        patience = self.config.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader, optimizer)
            val_loss, val_acc = self.evaluate(val_dataloader)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        return best_val_acc
    
    def save_model(self, model_name, learning_rate, best_val_acc, timestamp=None):
        """
        Save the trained model weights and hyperparameters.
        
        Args:
            model_name: Name of the model (e.g., 'bert', 'distilbert', 'roberta')
            learning_rate: Learning rate used for training
            best_val_acc: Best validation accuracy achieved
            timestamp: Optional timestamp string
            
        Returns:
            Dictionary with saved file paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model weights
        model_path = f"models/{model_name}_model_{timestamp}.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save hyperparameters and training info
        hyperparams = {
            'model_name': model_name,
            'learning_rate': learning_rate,
            'best_val_accuracy': best_val_acc,
            'timestamp': timestamp,
            'config': {
                'weight_decay': self.config.weight_decay,
                'early_stopping_patience': self.config.early_stopping_patience,
                'max_length': self.config.max_length,
                'batch_size': self.config.batch_sizes[0]
            }
        }
        
        hyperparams_path = f"models/{model_name}_hyperparams_{timestamp}.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Hyperparameters saved to {hyperparams_path}")
        
        return {
            'model_path': model_path,
            'hyperparams_path': hyperparams_path,
            'timestamp': timestamp
        }
    
    @staticmethod
    def load_model(model_class, model_path, hyperparams_path, device=None):
        """
        Load a saved model and its hyperparameters.
        
        Args:
            model_class: The model class to instantiate
            model_path: Path to the saved model weights
            hyperparams_path: Path to the saved hyperparameters
            device: Device to load the model on
            
        Returns:
            Tuple of (model, hyperparams_dict)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load hyperparameters
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        
        # Initialize model
        model = model_class()
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Hyperparameters loaded from {hyperparams_path}")
        
        return model, hyperparams

def create_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation, and testing."""
    
    train_dataset = PortalLegendsDataset(train_texts, train_labels, tokenizer)
    val_dataset = PortalLegendsDataset(val_texts, val_labels, tokenizer)
    test_dataset = PortalLegendsDataset(test_texts, test_labels, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader
