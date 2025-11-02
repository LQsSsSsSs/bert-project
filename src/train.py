"""
Training script for vulnerability classification model
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data_processor import VulnerabilityDataProcessor
from model import VulnerabilityBERTClassifier


class VulnerabilityDataset(Dataset):
    """PyTorch Dataset for vulnerability data"""
    
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        description = str(self.data.loc[idx, 'description'])
        type_label = int(self.data.loc[idx, 'type_label'])
        severity_label = int(self.data.loc[idx, 'severity_label'])
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'type_label': torch.tensor(type_label, dtype=torch.long),
            'severity_label': torch.tensor(severity_label, dtype=torch.long)
        }


class VulnerabilityTrainer:
    """Trainer for vulnerability classification model"""
    
    def __init__(self, config_path='config.json'):
        """Initialize trainer"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        model_config = self.config['model']
        self.tokenizer = BertTokenizer.from_pretrained(
            model_config['bert_model_name']
        )
        
        # Initialize model
        self.model = VulnerabilityBERTClassifier(config_path)
        self.model.to(self.device)
        
        self.max_length = model_config['max_length']
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_type_acc': [],
            'val_type_acc': [],
            'train_severity_acc': [],
            'val_severity_acc': []
        }
    
    def prepare_dataloaders(self, data_dict):
        """Prepare PyTorch DataLoaders"""
        batch_size = self.config['training']['batch_size']
        
        train_dataset = VulnerabilityDataset(
            data_dict['train'], self.tokenizer, self.max_length
        )
        val_dataset = VulnerabilityDataset(
            data_dict['val'], self.tokenizer, self.max_length
        )
        test_dataset = VulnerabilityDataset(
            data_dict['test'], self.tokenizer, self.max_length
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def configure_optimizer(self, train_loader):
        """Configure optimizer and learning rate scheduler"""
        training_config = self.config['training']
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        total_steps = len(train_loader) * training_config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        type_preds, type_labels = [], []
        severity_preds, severity_labels = [], []
        
        criterion = nn.CrossEntropyLoss()
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            type_label = batch['type_label'].to(self.device)
            severity_label = batch['severity_label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, task='both')
            
            # Calculate losses
            type_loss = criterion(outputs['type_logits'], type_label)
            severity_loss = criterion(outputs['severity_logits'], severity_label)
            loss = type_loss + severity_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            type_preds.extend(torch.argmax(outputs['type_logits'], dim=1).cpu().numpy())
            type_labels.extend(type_label.cpu().numpy())
            severity_preds.extend(torch.argmax(outputs['severity_logits'], dim=1).cpu().numpy())
            severity_labels.extend(severity_label.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        type_acc = accuracy_score(type_labels, type_preds)
        severity_acc = accuracy_score(severity_labels, severity_preds)
        
        return avg_loss, type_acc, severity_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        type_preds, type_labels = [], []
        severity_preds, severity_labels = [], []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                type_label = batch['type_label'].to(self.device)
                severity_label = batch['severity_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, task='both')
                
                # Calculate losses
                type_loss = criterion(outputs['type_logits'], type_label)
                severity_loss = criterion(outputs['severity_logits'], severity_label)
                loss = type_loss + severity_loss
                
                total_loss += loss.item()
                
                # Collect predictions
                type_preds.extend(torch.argmax(outputs['type_logits'], dim=1).cpu().numpy())
                type_labels.extend(type_label.cpu().numpy())
                severity_preds.extend(torch.argmax(outputs['severity_logits'], dim=1).cpu().numpy())
                severity_labels.extend(severity_label.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        type_acc = accuracy_score(type_labels, type_preds)
        severity_acc = accuracy_score(severity_labels, severity_preds)
        
        return avg_loss, type_acc, severity_acc
    
    def train(self, data_dict, output_dir='models'):
        """Complete training loop"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_dataloaders(data_dict)
        
        # Configure optimizer
        optimizer, scheduler = self.configure_optimizer(train_loader)
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_type_acc, train_severity_acc = self.train_epoch(
                train_loader, optimizer, scheduler
            )
            
            # Validate
            val_loss, val_type_acc, val_severity_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_type_acc'].append(train_type_acc)
            self.history['val_type_acc'].append(val_type_acc)
            self.history['train_severity_acc'].append(train_severity_acc)
            self.history['val_severity_acc'].append(val_severity_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Type Acc: {train_type_acc:.4f}, Val Type Acc: {val_type_acc:.4f}")
            print(f"Train Severity Acc: {train_severity_acc:.4f}, Val Severity Acc: {val_severity_acc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = os.path.join(output_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        test_loss, test_type_acc, test_severity_acc = self.validate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Type Accuracy: {test_type_acc:.4f}")
        print(f"Test Severity Accuracy: {test_severity_acc:.4f}")
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training history
        self.plot_history(output_dir)
        
        return self.history
    
    def plot_history(self, output_dir):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Type Accuracy
        axes[0, 1].plot(self.history['train_type_acc'], label='Train')
        axes[0, 1].plot(self.history['val_type_acc'], label='Validation')
        axes[0, 1].set_title('Vulnerability Type Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Severity Accuracy
        axes[1, 0].plot(self.history['train_severity_acc'], label='Train')
        axes[1, 0].plot(self.history['val_severity_acc'], label='Validation')
        axes[1, 0].set_title('Severity Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(plot_path)
        print(f"Saved training plots to {plot_path}")


def main():
    """Main training function"""
    # Process data
    processor = VulnerabilityDataProcessor()
    data_dict = processor.prepare_data('data/vulnerabilities.csv')
    
    # Train model
    trainer = VulnerabilityTrainer()
    trainer.train(data_dict)


if __name__ == '__main__':
    main()
