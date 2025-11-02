# Examples and Use Cases

## Quick Start Examples

### Example 1: Basic Prediction

```python
from src.model import VulnerabilityPredictor

# Load trained model
predictor = VulnerabilityPredictor('models/best_model.pt')

# Make prediction
description = "SQL injection vulnerability in user authentication"
result = predictor.predict(description)

print(f"Type: {result['type_label']} (confidence: {result['type_confidence']:.2%})")
print(f"Severity: {result['severity_label']} (confidence: {result['severity_confidence']:.2%})")
```

### Example 2: Batch Prediction

```python
from src.model import VulnerabilityPredictor

predictor = VulnerabilityPredictor('models/best_model.pt')

descriptions = [
    "Buffer overflow in image parser",
    "XSS vulnerability in search feature",
    "Weak password hashing algorithm"
]

results = predictor.batch_predict(descriptions)

for desc, result in zip(descriptions, results):
    print(f"\nDescription: {desc}")
    print(f"Type: {result['type_label']}")
    print(f"Severity: {result['severity_label']}")
```

### Example 3: Custom Training

```python
from src.data_processor import VulnerabilityDataProcessor
from src.train import VulnerabilityTrainer

# Prepare data
processor = VulnerabilityDataProcessor('config.json')
data_dict = processor.prepare_data('data/my_vulnerabilities.csv')

# Train model
trainer = VulnerabilityTrainer('config.json')
history = trainer.train(data_dict, output_dir='models')

print(f"Best validation loss: {min(history['val_loss'])}")
```

### Example 4: Model Evaluation

```python
from src.evaluate import ModelEvaluator
from src.data_processor import VulnerabilityDataProcessor

# Load data
processor = VulnerabilityDataProcessor()
data_dict = processor.prepare_data('data/vulnerabilities.csv')

# Evaluate
evaluator = ModelEvaluator('models/best_model.pt')
results, *_ = evaluator.evaluate(data_dict['test'])

print(f"Type Accuracy: {results['type_accuracy']:.4f}")
print(f"Severity Accuracy: {results['severity_accuracy']:.4f}")
```

## Real-World Use Cases

### Use Case 1: Automated Vulnerability Triage

**Scenario**: Security team receives hundreds of vulnerability reports daily.

**Solution**: 
```python
import pandas as pd
from src.model import VulnerabilityPredictor

# Load vulnerability reports
reports = pd.read_csv('daily_reports.csv')

# Initialize predictor
predictor = VulnerabilityPredictor('models/best_model.pt')

# Process all reports
results = []
for _, row in reports.iterrows():
    prediction = predictor.predict(row['description'])
    results.append({
        'id': row['id'],
        'description': row['description'],
        'predicted_type': prediction['type_label'],
        'predicted_severity': prediction['severity_label'],
        'confidence': prediction['severity_confidence']
    })

# Save results
results_df = pd.DataFrame(results)

# Prioritize critical and high severity
critical_high = results_df[results_df['predicted_severity'].isin(['Critical', 'High'])]
critical_high = critical_high.sort_values('confidence', ascending=False)

print(f"Found {len(critical_high)} critical/high severity vulnerabilities")
critical_high.to_csv('priority_vulnerabilities.csv', index=False)
```

### Use Case 2: Security Metrics Dashboard

**Scenario**: Track vulnerability trends over time.

```python
import pandas as pd
from src.model import VulnerabilityPredictor
import matplotlib.pyplot as plt

# Load historical data
vulns = pd.read_csv('historical_vulnerabilities.csv')
predictor = VulnerabilityPredictor('models/best_model.pt')

# Classify all vulnerabilities
vulns['predicted_type'] = vulns['description'].apply(
    lambda x: predictor.predict(x)['type_label']
)
vulns['predicted_severity'] = vulns['description'].apply(
    lambda x: predictor.predict(x)['severity_label']
)

# Generate statistics
type_counts = vulns['predicted_type'].value_counts()
severity_counts = vulns['predicted_severity'].value_counts()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

type_counts.plot(kind='bar', ax=axes[0])
axes[0].set_title('Vulnerability Types Distribution')
axes[0].set_xlabel('Type')
axes[0].set_ylabel('Count')

severity_counts.plot(kind='bar', ax=axes[1], color=['green', 'yellow', 'orange', 'red'])
axes[1].set_title('Severity Distribution')
axes[1].set_xlabel('Severity')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('vulnerability_dashboard.png')
```

### Use Case 3: Integration with Bug Bounty Platform

**Scenario**: Automatically classify and prioritize bug bounty submissions.

```python
from flask import Flask, request, jsonify
from src.model import VulnerabilityPredictor

app = Flask(__name__)
predictor = VulnerabilityPredictor('models/best_model.pt')

@app.route('/classify', methods=['POST'])
def classify_vulnerability():
    data = request.json
    description = data.get('description', '')
    
    if not description:
        return jsonify({'error': 'No description provided'}), 400
    
    # Make prediction
    result = predictor.predict(description)
    
    # Calculate bounty based on severity
    bounty_map = {
        'Low': 100,
        'Medium': 500,
        'High': 2000,
        'Critical': 5000
    }
    
    response = {
        'vulnerability_type': result['type_label'],
        'severity': result['severity_label'],
        'type_confidence': result['type_confidence'],
        'severity_confidence': result['severity_confidence'],
        'suggested_bounty': bounty_map[result['severity_label']]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Use Case 4: Continuous Training Pipeline

**Scenario**: Regularly retrain model with new vulnerability data.

```python
import os
from datetime import datetime
from src.data_processor import VulnerabilityDataProcessor
from src.train import VulnerabilityTrainer
from src.evaluate import ModelEvaluator

def retrain_model(data_path, min_accuracy=0.80):
    """Retrain model and deploy if accuracy improves"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'models/version_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print(f"Loading data from {data_path}...")
    processor = VulnerabilityDataProcessor()
    data_dict = processor.prepare_data(data_path)
    
    # Train new model
    print("Training new model...")
    trainer = VulnerabilityTrainer()
    trainer.train(data_dict, output_dir=output_dir)
    
    # Evaluate
    print("Evaluating new model...")
    evaluator = ModelEvaluator(f'{output_dir}/best_model.pt')
    results, *_ = evaluator.evaluate(data_dict['test'])
    
    avg_accuracy = (results['type_accuracy'] + results['severity_accuracy']) / 2
    
    print(f"\nNew model average accuracy: {avg_accuracy:.4f}")
    
    # Deploy if meets threshold
    if avg_accuracy >= min_accuracy:
        print("✓ Model meets accuracy threshold")
        print(f"Deploying model from {output_dir}")
        
        # Copy to production location
        import shutil
        shutil.copy(
            f'{output_dir}/best_model.pt',
            'models/production_model.pt'
        )
        print("✓ Model deployed successfully")
        return True
    else:
        print(f"✗ Model accuracy {avg_accuracy:.4f} below threshold {min_accuracy:.4f}")
        print("Keeping current production model")
        return False

# Run weekly retraining
if __name__ == '__main__':
    retrain_model('data/vulnerabilities.csv', min_accuracy=0.80)
```

### Use Case 5: Export Predictions for Reporting

**Scenario**: Generate weekly security reports.

```python
import pandas as pd
from datetime import datetime, timedelta
from src.model import VulnerabilityPredictor

def generate_weekly_report(start_date, end_date):
    """Generate vulnerability report for date range"""
    
    # Load vulnerabilities
    vulns = pd.read_csv('vulnerabilities.csv')
    vulns['date'] = pd.to_datetime(vulns['date'])
    
    # Filter by date range
    mask = (vulns['date'] >= start_date) & (vulns['date'] <= end_date)
    week_vulns = vulns[mask]
    
    if len(week_vulns) == 0:
        print("No vulnerabilities in date range")
        return
    
    # Classify
    predictor = VulnerabilityPredictor('models/best_model.pt')
    
    predictions = []
    for _, row in week_vulns.iterrows():
        result = predictor.predict(row['description'])
        predictions.append({
            'Date': row['date'],
            'Description': row['description'][:100] + '...',
            'Type': result['type_label'],
            'Severity': result['severity_label'],
            'Type Confidence': f"{result['type_confidence']:.2%}",
            'Severity Confidence': f"{result['severity_confidence']:.2%}"
        })
    
    # Create report
    report_df = pd.DataFrame(predictions)
    
    # Summary statistics
    print(f"\nWeekly Security Report ({start_date} to {end_date})")
    print("="*60)
    print(f"\nTotal Vulnerabilities: {len(report_df)}")
    print("\nBy Type:")
    print(report_df['Type'].value_counts())
    print("\nBy Severity:")
    print(report_df['Severity'].value_counts())
    
    # Save report
    filename = f"security_report_{start_date.strftime('%Y%m%d')}.csv"
    report_df.to_csv(filename, index=False)
    print(f"\n✓ Report saved to {filename}")

# Generate report for last week
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
generate_weekly_report(start_date, end_date)
```

## Sample Data Format

### Input CSV Format

```csv
description,vulnerability_type,severity
"SQL injection in login endpoint allows database access",SQL Injection,Critical
"XSS vulnerability enables session hijacking","Cross-Site Scripting (XSS)",High
"Buffer overflow in packet parser",Buffer Overflow,Critical
```

### Prediction Output Format

```json
{
  "type_pred": 0,
  "type_label": "SQL Injection",
  "type_confidence": 0.9534,
  "severity_pred": 3,
  "severity_label": "Critical",
  "severity_confidence": 0.8945
}
```

## Command Line Examples

### Training
```bash
# Basic training
cd src && python train.py

# View training progress
tail -f logs/training.log
```

### Prediction
```bash
# Single prediction
cd src && python predict.py \
  --description "SQL injection in user input" \
  --model ../models/best_model.pt

# Prediction with custom config
cd src && python predict.py \
  --description "XSS in comment field" \
  --model ../models/best_model.pt \
  --config ../custom_config.json
```

### Evaluation
```bash
# Evaluate on test set
cd src && python evaluate.py

# Evaluate with custom data
cd src && python evaluate.py \
  --data ../data/custom_test.csv \
  --model ../models/best_model.pt
```

## Tips and Best Practices

1. **Data Quality**: Ensure your training data is clean and well-labeled
2. **Class Balance**: Try to balance classes for better performance
3. **Hyperparameter Tuning**: Experiment with learning rate and batch size
4. **Regular Retraining**: Update model with new vulnerability data
5. **Confidence Thresholds**: Set minimum confidence for automatic classification
6. **Human Review**: Always have security experts review high-confidence predictions
7. **Monitoring**: Track model performance over time
8. **Version Control**: Keep track of model versions and their performance
