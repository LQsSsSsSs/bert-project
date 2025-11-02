"""
Evaluation script for trained model
"""

import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from data_processor import VulnerabilityDataProcessor
from train import VulnerabilityDataset
from model import VulnerabilityBERTClassifier
from transformers import BertTokenizer


class ModelEvaluator:
    """Evaluator for vulnerability classification model"""
    
    def __init__(self, model_path, config_path='config.json'):
        """Initialize evaluator"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = VulnerabilityBERTClassifier(config_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        model_config = self.config['model']
        self.tokenizer = BertTokenizer.from_pretrained(
            model_config['bert_model_name']
        )
        self.max_length = model_config['max_length']
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        # Create dataset and dataloader
        test_dataset = VulnerabilityDataset(
            test_data, self.tokenizer, self.max_length
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        type_preds, type_labels = [], []
        severity_preds, severity_labels = [], []
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                type_label = batch['type_label']
                severity_label = batch['severity_label']
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, task='both')
                
                # Collect predictions
                type_preds.extend(
                    torch.argmax(outputs['type_logits'], dim=1).cpu().numpy()
                )
                type_labels.extend(type_label.numpy())
                severity_preds.extend(
                    torch.argmax(outputs['severity_logits'], dim=1).cpu().numpy()
                )
                severity_labels.extend(severity_label.numpy())
        
        # Calculate metrics
        results = self.calculate_metrics(
            type_labels, type_preds,
            severity_labels, severity_preds
        )
        
        return results, type_labels, type_preds, severity_labels, severity_preds
    
    def calculate_metrics(self, type_labels, type_preds, severity_labels, severity_preds):
        """Calculate evaluation metrics"""
        results = {}
        
        # Type classification metrics
        results['type_accuracy'] = accuracy_score(type_labels, type_preds)
        type_metrics = precision_recall_fscore_support(
            type_labels, type_preds, average='weighted', zero_division=0
        )
        results['type_precision'] = type_metrics[0]
        results['type_recall'] = type_metrics[1]
        results['type_f1'] = type_metrics[2]
        
        # Severity classification metrics
        results['severity_accuracy'] = accuracy_score(severity_labels, severity_preds)
        severity_metrics = precision_recall_fscore_support(
            severity_labels, severity_preds, average='weighted', zero_division=0
        )
        results['severity_precision'] = severity_metrics[0]
        results['severity_recall'] = severity_metrics[1]
        results['severity_f1'] = severity_metrics[2]
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print("\nVulnerability Type Classification:")
        print(f"  Accuracy:  {results['type_accuracy']:.4f}")
        print(f"  Precision: {results['type_precision']:.4f}")
        print(f"  Recall:    {results['type_recall']:.4f}")
        print(f"  F1-Score:  {results['type_f1']:.4f}")
        
        print("\nSeverity Classification:")
        print(f"  Accuracy:  {results['severity_accuracy']:.4f}")
        print(f"  Precision: {results['severity_precision']:.4f}")
        print(f"  Recall:    {results['severity_recall']:.4f}")
        print(f"  F1-Score:  {results['severity_f1']:.4f}")
        
        print("="*60)
    
    def plot_confusion_matrices(self, type_labels, type_preds, 
                                severity_labels, severity_preds, output_dir='models'):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Type confusion matrix
        type_cm = confusion_matrix(type_labels, type_preds)
        sns.heatmap(
            type_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.config['vulnerability_types'],
            yticklabels=self.config['vulnerability_types'],
            ax=axes[0]
        )
        axes[0].set_title('Vulnerability Type Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(axes[0].yaxis.get_majorticklabels(), rotation=0)
        
        # Severity confusion matrix
        severity_cm = confusion_matrix(severity_labels, severity_preds)
        sns.heatmap(
            severity_cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=self.config['severity_levels'],
            yticklabels=self.config['severity_levels'],
            ax=axes[1]
        )
        axes[1].set_title('Severity Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plot_path = f'{output_dir}/confusion_matrices.png'
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"\nSaved confusion matrices to {plot_path}")
    
    def generate_classification_reports(self, type_labels, type_preds,
                                       severity_labels, severity_preds):
        """Generate detailed classification reports"""
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*60)
        
        print("\nVulnerability Type Classification Report:")
        print(classification_report(
            type_labels, type_preds,
            target_names=self.config['vulnerability_types'],
            zero_division=0
        ))
        
        print("\nSeverity Classification Report:")
        print(classification_report(
            severity_labels, severity_preds,
            target_names=self.config['severity_levels'],
            zero_division=0
        ))


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate vulnerability classification model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.pt',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/sample_vulnerabilities.csv',
        help='Path to test data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Load and prepare data
    print("Loading data...")
    processor = VulnerabilityDataProcessor(args.config)
    data_dict = processor.prepare_data(args.data)
    
    # Evaluate model
    evaluator = ModelEvaluator(args.model, args.config)
    results, type_labels, type_preds, severity_labels, severity_preds = evaluator.evaluate(
        data_dict['test']
    )
    
    # Print results
    evaluator.print_results(results)
    evaluator.generate_classification_reports(
        type_labels, type_preds,
        severity_labels, severity_preds
    )
    evaluator.plot_confusion_matrices(
        type_labels, type_preds,
        severity_labels, severity_preds
    )
    
    # Save results
    with open('models/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved evaluation results to models/evaluation_results.json")


if __name__ == '__main__':
    main()
