"""
BERT-based Vulnerability Classification and Severity Prediction System
"""

__version__ = "1.0.0"
__author__ = "Vulnerability Analysis Team"

from .model import VulnerabilityBERTClassifier, VulnerabilityPredictor
from .data_processor import VulnerabilityDataProcessor
from .train import VulnerabilityTrainer

__all__ = [
    'VulnerabilityBERTClassifier',
    'VulnerabilityPredictor',
    'VulnerabilityDataProcessor',
    'VulnerabilityTrainer'
]
