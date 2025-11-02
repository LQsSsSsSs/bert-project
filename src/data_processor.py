"""
BERT-based Vulnerability Classification and Severity Prediction
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json


class VulnerabilityDataProcessor:
    """Process vulnerability data for BERT model training"""
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize data processor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.type_encoder = LabelEncoder()
        self.severity_encoder = LabelEncoder()
        
        # Fit encoders with predefined labels
        self.type_encoder.fit(self.config['vulnerability_types'])
        self.severity_encoder.fit(self.config['severity_levels'])
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load vulnerability data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with vulnerability data
        """
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['description', 'vulnerability_type', 'severity']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess vulnerability description text
        
        Args:
            text: Raw text description
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Basic preprocessing
        text = str(text).strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical labels to numeric values
        
        Args:
            df: DataFrame with vulnerability data
            
        Returns:
            DataFrame with encoded labels
        """
        df = df.copy()
        
        # Preprocess text
        df['description'] = df['description'].apply(self.preprocess_text)
        
        # Encode labels
        df['type_label'] = self.type_encoder.transform(df['vulnerability_type'])
        df['severity_label'] = self.severity_encoder.transform(df['severity'])
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: DataFrame with vulnerability data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_ratio = self.config['data']['train_split']
        val_ratio = self.config['data']['val_split']
        test_ratio = self.config['data']['test_split']
        random_seed = self.config['data']['random_seed']
        
        # First split: train and temp (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio),
            random_state=random_seed,
            stratify=df['severity_label']
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=random_seed,
            stratify=temp_df['severity_label']
        )
        
        return train_df, val_df, test_df
    
    def prepare_data(self, filepath: str) -> Dict:
        """
        Complete data preparation pipeline
        
        Args:
            filepath: Path to data file
            
        Returns:
            Dictionary containing train, val, and test data
        """
        # Load data
        df = self.load_data(filepath)
        
        # Encode labels
        df = self.encode_labels(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        print(f"Data preparation complete:")
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples: {len(val_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'type_encoder': self.type_encoder,
            'severity_encoder': self.severity_encoder
        }
    
    def decode_type_label(self, label: int) -> str:
        """Decode vulnerability type label to text"""
        return self.type_encoder.inverse_transform([label])[0]
    
    def decode_severity_label(self, label: int) -> str:
        """Decode severity label to text"""
        return self.severity_encoder.inverse_transform([label])[0]
