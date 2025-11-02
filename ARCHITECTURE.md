# Architecture Documentation

## System Overview

The BERT-based Vulnerability Classification and Severity Prediction system is designed to automatically analyze vulnerability descriptions and predict both the type of vulnerability and its severity level.

## Architecture Components

### 1. Data Processing Layer (`data_processor.py`)

**VulnerabilityDataProcessor**
- Loads vulnerability data from CSV files
- Preprocesses text descriptions
- Encodes categorical labels
- Splits data into train/validation/test sets
- Handles label encoding/decoding

### 2. Model Layer (`model.py`)

**VulnerabilityBERTClassifier**
- Multi-task neural network based on BERT
- Dual classification heads:
  - Vulnerability type classifier (5 classes)
  - Severity classifier (4 classes)
- Uses BERT's [CLS] token representation for classification
- Supports task-specific inference

**VulnerabilityPredictor**
- Inference wrapper for trained models
- Handles tokenization and prediction
- Returns predictions with confidence scores
- Supports batch prediction

### 3. Training Layer (`train.py`)

**VulnerabilityDataset**
- PyTorch Dataset implementation
- Handles tokenization and encoding
- Batch preparation for training

**VulnerabilityTrainer**
- Complete training pipeline
- Implements training and validation loops
- Model checkpointing
- Training history tracking
- Visualization generation

### 4. Evaluation Layer (`evaluate.py`)

**ModelEvaluator**
- Comprehensive model evaluation
- Metrics calculation (accuracy, precision, recall, F1)
- Classification reports
- Confusion matrix generation
- Visualization of results

## Data Flow

```
1. Input: Vulnerability Description (Text)
   ↓
2. Tokenization (BERT Tokenizer)
   ↓
3. BERT Encoder (Pre-trained)
   ↓
4. [CLS] Token Representation (768-dim vector)
   ↓
5. Dropout Layer
   ↓
6. Parallel Classification Heads
   ├─→ Type Classifier (Linear Layer) → Type Prediction
   └─→ Severity Classifier (Linear Layer) → Severity Prediction
```

## Model Architecture

```python
VulnerabilityBERTClassifier(
  bert: BertModel(
    embeddings: BertEmbeddings(...)
    encoder: BertEncoder(
      layer: ModuleList(
        (0-11): 12 x BertLayer(...)
      )
    )
    pooler: BertPooler(...)
  )
  dropout: Dropout(p=0.1)
  type_classifier: Linear(in_features=768, out_features=5)
  severity_classifier: Linear(in_features=768, out_features=4)
)
```

## Training Process

### Initialization
1. Load configuration from `config.json`
2. Initialize BERT model with pre-trained weights
3. Add classification heads
4. Set up optimizer (AdamW) and learning rate scheduler

### Training Loop
For each epoch:
1. **Training Phase**
   - Forward pass through BERT and classifiers
   - Calculate loss for both tasks (cross-entropy)
   - Combined loss = type_loss + severity_loss
   - Backward pass and gradient update
   - Gradient clipping for stability

2. **Validation Phase**
   - Evaluate on validation set
   - Calculate metrics (accuracy, loss)
   - Save best model checkpoint

3. **Monitoring**
   - Track training/validation losses
   - Track accuracy for both tasks
   - Generate plots and logs

### Post-Training
1. Evaluate on test set
2. Generate classification reports
3. Create confusion matrices
4. Save final model and metrics

## Configuration

The system uses a JSON configuration file with the following structure:

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
    "num_epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0
  },
  "data": {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42
  },
  "vulnerability_types": [...],
  "severity_levels": [...]
}
```

## Key Design Decisions

### 1. Multi-Task Learning
- **Why**: Vulnerability type and severity are related tasks
- **Benefit**: Shared BERT representations improve both tasks
- **Trade-off**: Slightly more complex training logic

### 2. BERT-base-uncased
- **Why**: Good balance of performance and resource requirements
- **Alternative**: bert-large for better performance (more resources)
- **Customization**: Can use domain-specific BERT variants

### 3. Fixed Max Length (512 tokens)
- **Why**: BERT's maximum sequence length
- **Benefit**: Handles most vulnerability descriptions
- **Trade-off**: Very long descriptions are truncated

### 4. Combined Loss Function
- **Why**: Simple and effective for multi-task learning
- **Alternative**: Weighted combination (if one task is more important)
- **Implementation**: loss = type_loss + severity_loss

## Performance Considerations

### Memory Usage
- BERT-base: ~440MB
- Batch size 16: ~2-4GB GPU memory
- Reduce batch size if out of memory

### Training Time
- Sample dataset (50 examples): ~5-10 minutes on CPU
- Full dataset (1000+ examples): ~1-2 hours on GPU
- Scales linearly with dataset size

### Inference Speed
- Single prediction: ~100-200ms on CPU
- Batch prediction: ~10-20ms per sample on GPU
- Can be optimized with ONNX or quantization

## Extension Points

### Adding New Vulnerability Types
1. Update `vulnerability_types` in `config.json`
2. Update `num_classification_classes` accordingly
3. Ensure training data includes new types
4. Retrain model

### Adding New Features
1. **Attention Visualization**: Add attention weight extraction
2. **Ensemble Methods**: Combine multiple models
3. **Active Learning**: Iterative training with human feedback
4. **Explainability**: Add LIME or SHAP integration

### Custom BERT Models
1. Change `bert_model_name` in config
2. Options: `bert-large-uncased`, `roberta-base`, domain-specific models
3. Adjust `max_length` and `batch_size` accordingly

## Deployment Considerations

### Production Deployment
1. **Model Serving**: Use FastAPI or Flask
2. **Containerization**: Docker for easy deployment
3. **Scaling**: Load balancing for multiple instances
4. **Monitoring**: Track prediction latency and accuracy

### Model Updates
1. Version trained models
2. A/B testing for new models
3. Gradual rollout to production
4. Fallback to previous version if issues

## Security Considerations

1. **Input Validation**: Sanitize input descriptions
2. **Rate Limiting**: Prevent abuse of prediction API
3. **Model Security**: Protect model files from unauthorized access
4. **Data Privacy**: Ensure compliance with data protection regulations
