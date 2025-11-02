# BERT-based Vulnerability Classification and Severity Prediction

基于BERT的漏洞描述自动分类与严重性预测系统

## Overview

This project implements a deep learning system using BERT (Bidirectional Encoder Representations from Transformers) for automatic classification of vulnerability descriptions and prediction of their severity levels. The system can:

1. **Classify vulnerability types** into categories such as:
   - SQL Injection
   - Cross-Site Scripting (XSS)
   - Buffer Overflow
   - Authentication Bypass
   - Other

2. **Predict severity levels**:
   - Low
   - Medium
   - High
   - Critical

## Features

- **Multi-task Learning**: Simultaneously learns to classify vulnerability types and predict severity
- **Pre-trained BERT**: Leverages BERT's language understanding capabilities
- **Fine-tuning Support**: Can be fine-tuned on custom vulnerability datasets
- **Easy-to-use API**: Simple interfaces for training, evaluation, and prediction
- **Comprehensive Metrics**: Detailed evaluation with accuracy, precision, recall, F1-score, and confusion matrices
- **Visualization**: Training history plots and confusion matrices

## Project Structure

```
.
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data directory
│   └── sample_vulnerabilities.csv  # Sample dataset
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── data_processor.py      # Data preprocessing utilities
│   ├── model.py               # BERT model implementation
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction script
│   └── evaluate.py            # Evaluation script
└── models/                     # Directory for trained models (created during training)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LQsSsSsSs/-
cd -
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Prepare your vulnerability data in CSV format with the following columns:
- `description`: Vulnerability description text
- `vulnerability_type`: Type of vulnerability
- `severity`: Severity level

Example format (see `data/sample_vulnerabilities.csv`):
```csv
description,vulnerability_type,severity
"SQL injection vulnerability allows attackers to execute arbitrary SQL commands",SQL Injection,Critical
"Cross-site scripting vulnerability in comment section","Cross-Site Scripting (XSS)",High
```

### 2. Training

Train the model on your dataset:

```bash
cd src
python train.py
```

The training script will:
- Load and preprocess data from `data/vulnerabilities.csv`
- Split data into train/validation/test sets
- Train the BERT-based model
- Save the best model to `models/best_model.pt`
- Generate training history plots
- Evaluate on the test set

**Note**: For the sample dataset, you can use:
```bash
cd src
python train.py
# Make sure to update the data path in train.py to use sample_vulnerabilities.csv
```

### 3. Prediction

Make predictions on new vulnerability descriptions:

```bash
cd src
python predict.py --description "SQL injection in login form permits database extraction"
```

Output:
```
==================================================
PREDICTION RESULTS
==================================================

Input Description:
  SQL injection in login form permits database extraction

Vulnerability Type:
  SQL Injection
  Confidence: 95.32%

Severity:
  Critical
  Confidence: 89.45%
==================================================
```

### 4. Evaluation

Evaluate the trained model:

```bash
cd src
python evaluate.py
```

This will generate:
- Accuracy, precision, recall, and F1-score for both tasks
- Detailed classification reports
- Confusion matrices
- Evaluation results saved to `models/evaluation_results.json`

## Configuration

Edit `config.json` to customize:

- **Model settings**: BERT model name, max length, dropout
- **Training parameters**: Batch size, learning rate, epochs
- **Data splits**: Train/validation/test ratios
- **Class labels**: Vulnerability types and severity levels

Example configuration:
```json
{
  "model": {
    "bert_model_name": "bert-base-uncased",
    "max_length": 512,
    "num_classification_classes": 5,
    "num_severity_classes": 4,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10
  }
}
```

## Model Architecture

The system uses a BERT-based multi-task learning architecture:

```
Input Text
    ↓
BERT Encoder
    ↓
[CLS] Token Representation
    ↓
Dropout
    ↓
    ├─→ Vulnerability Type Classifier → Type Prediction
    └─→ Severity Classifier → Severity Prediction
```

## API Reference

### VulnerabilityDataProcessor

```python
from data_processor import VulnerabilityDataProcessor

processor = VulnerabilityDataProcessor('config.json')
data_dict = processor.prepare_data('data/vulnerabilities.csv')
```

### VulnerabilityTrainer

```python
from train import VulnerabilityTrainer

trainer = VulnerabilityTrainer('config.json')
trainer.train(data_dict, output_dir='models')
```

### VulnerabilityPredictor

```python
from model import VulnerabilityPredictor

predictor = VulnerabilityPredictor('models/best_model.pt', 'config.json')
result = predictor.predict("Your vulnerability description here")
```

## Performance

The model's performance depends on the quality and size of the training data. On the sample dataset, typical results include:

- **Vulnerability Type Classification**: 85-95% accuracy
- **Severity Prediction**: 80-90% accuracy

For better results:
- Use a larger, more diverse dataset
- Fine-tune hyperparameters
- Increase training epochs
- Use domain-specific BERT variants

## Requirements

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

See `requirements.txt` for complete list.

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `config.json`
- Reduce max sequence length
- Use gradient accumulation

### Poor Performance
- Collect more training data
- Balance class distributions
- Increase training epochs
- Try different BERT variants (e.g., `bert-large-uncased`)

### Slow Training
- Use GPU if available
- Reduce max sequence length
- Increase batch size (if memory allows)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vulnerability_bert_classifier,
  title = {BERT-based Vulnerability Classification and Severity Prediction},
  author = {Vulnerability Analysis Team},
  year = {2025},
  url = {https://github.com/LQsSsSsSs/-}
}
```

## Acknowledgments

- Built on top of Hugging Face Transformers
- Uses pre-trained BERT models from Google Research

## Contact

For questions or issues, please open an issue on GitHub.