# Quick Reference Guide

## Installation

```bash
# Clone repository
git clone https://github.com/LQsSsSsSs/-
cd -

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

## Quick Start

```bash
# Run demo
python demo.py

# Train model (requires data at data/vulnerabilities.csv)
cd src
python train.py

# Make prediction
python predict.py --description "SQL injection in login form"

# Evaluate model
python evaluate.py
```

## Python API

### Data Processing

```python
from src.data_processor import VulnerabilityDataProcessor

processor = VulnerabilityDataProcessor('config.json')
data = processor.prepare_data('data/vulnerabilities.csv')
```

### Training

```python
from src.train import VulnerabilityTrainer

trainer = VulnerabilityTrainer('config.json')
history = trainer.train(data, output_dir='models')
```

### Prediction

```python
from src.model import VulnerabilityPredictor

predictor = VulnerabilityPredictor('models/best_model.pt')
result = predictor.predict("Your vulnerability description")

print(f"Type: {result['type_label']}")
print(f"Severity: {result['severity_label']}")
```

### Evaluation

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator('models/best_model.pt')
results, *_ = evaluator.evaluate(test_data)
```

## Command Line Interface

### Training
```bash
cd src
python train.py
```

### Prediction
```bash
cd src
python predict.py \
  --model ../models/best_model.pt \
  --description "Your vulnerability description" \
  --config ../config.json
```

### Evaluation
```bash
cd src
python evaluate.py \
  --model ../models/best_model.pt \
  --data ../data/test.csv \
  --config ../config.json
```

## Configuration

Edit `config.json` to customize:

```json
{
  "model": {
    "bert_model_name": "bert-base-uncased",
    "max_length": 512,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10
  }
}
```

## Data Format

CSV file with columns:
- `description`: Text description of vulnerability
- `vulnerability_type`: One of the predefined types
- `severity`: One of: Low, Medium, High, Critical

Example:
```csv
description,vulnerability_type,severity
"SQL injection allows database access",SQL Injection,Critical
"XSS in search field","Cross-Site Scripting (XSS)",High
```

## Common Tasks

### Add New Vulnerability Type
1. Edit `config.json`:
   - Add to `vulnerability_types` list
   - Update `num_classification_classes`
2. Add training data with new type
3. Retrain model

### Adjust Model Size
```json
{
  "model": {
    "bert_model_name": "bert-large-uncased"  // Larger model
  }
}
```

### Reduce Memory Usage
```json
{
  "training": {
    "batch_size": 8  // Smaller batch size
  },
  "model": {
    "max_length": 256  // Shorter sequences
  }
}
```

### Speed Up Training
- Use GPU if available (automatically detected)
- Increase batch size if memory allows
- Reduce max_length
- Reduce num_epochs for quick testing

## Troubleshooting

### Out of Memory
```json
{"training": {"batch_size": 4}}
```

### Slow Training
- Check if GPU is being used: `torch.cuda.is_available()`
- Reduce `max_length` in config
- Use `bert-base` instead of `bert-large`

### Poor Accuracy
- Increase `num_epochs`
- Collect more training data
- Balance class distributions
- Try different learning rate

## File Locations

- **Models**: `models/best_model.pt`
- **Training history**: `models/training_history.json`
- **Plots**: `models/training_history.png`, `models/confusion_matrices.png`
- **Logs**: `logs/`
- **Config**: `config.json`
- **Data**: `data/`

## Environment Variables

None required. Optional:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer warnings
```

## Performance Tips

1. **GPU Usage**: Automatically uses GPU if available
2. **Batch Size**: Larger = faster but more memory
3. **Workers**: Set `num_workers` in DataLoader for parallel loading
4. **Mixed Precision**: Add for faster training (requires modification)
5. **Model Caching**: BERT models are cached after first download

## Links

- [README.md](README.md) - Full documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- [EXAMPLES.md](EXAMPLES.md) - Use cases
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

## Support

- Open an issue on GitHub
- Check documentation files
- Run `python verify_setup.py` to diagnose setup issues
