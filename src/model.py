"""
BERT-based model for vulnerability classification and severity prediction
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import json


class VulnerabilityBERTClassifier(nn.Module):
    """BERT-based classifier for vulnerability type and severity"""
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize BERT classifier
        
        Args:
            config_path: Path to configuration file
        """
        super(VulnerabilityBERTClassifier, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        model_config = self.config['model']
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_config['bert_model_name'])
        
        # Dropout layer
        self.dropout = nn.Dropout(model_config['dropout'])
        
        # Classification heads
        hidden_size = self.bert.config.hidden_size
        
        # Vulnerability type classifier
        self.type_classifier = nn.Linear(
            hidden_size, 
            model_config['num_classification_classes']
        )
        
        # Severity classifier
        self.severity_classifier = nn.Linear(
            hidden_size,
            model_config['num_severity_classes']
        )
    
    def forward(self, input_ids, attention_mask, task='both'):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task: Task to perform ('type', 'severity', or 'both')
            
        Returns:
            Dictionary with type_logits and/or severity_logits
        """
        # Get BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        result = {}
        
        if task in ['type', 'both']:
            result['type_logits'] = self.type_classifier(pooled_output)
        
        if task in ['severity', 'both']:
            result['severity_logits'] = self.severity_classifier(pooled_output)
        
        return result


class VulnerabilityPredictor:
    """Predictor for trained vulnerability classification model"""
    
    def __init__(self, model_path: str, config_path: str = 'config.json'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model file
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        model_config = self.config['model']
        self.tokenizer = BertTokenizer.from_pretrained(
            model_config['bert_model_name']
        )
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VulnerabilityBERTClassifier(config_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = model_config['max_length']
    
    def predict(self, description: str):
        """
        Predict vulnerability type and severity
        
        Args:
            description: Vulnerability description text
            
        Returns:
            Dictionary with predicted type and severity
        """
        # Tokenize input
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, task='both')
        
        # Get predictions
        type_pred = torch.argmax(outputs['type_logits'], dim=1).item()
        severity_pred = torch.argmax(outputs['severity_logits'], dim=1).item()
        
        # Get probabilities
        type_probs = torch.softmax(outputs['type_logits'], dim=1)[0].cpu().numpy()
        severity_probs = torch.softmax(outputs['severity_logits'], dim=1)[0].cpu().numpy()
        
        return {
            'type_pred': type_pred,
            'type_label': self.config['vulnerability_types'][type_pred],
            'type_confidence': float(type_probs[type_pred]),
            'severity_pred': severity_pred,
            'severity_label': self.config['severity_levels'][severity_pred],
            'severity_confidence': float(severity_probs[severity_pred])
        }
    
    def batch_predict(self, descriptions: list):
        """
        Predict for multiple descriptions
        
        Args:
            descriptions: List of vulnerability descriptions
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for desc in descriptions:
            results.append(self.predict(desc))
        return results
