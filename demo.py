"""
Demo script showing basic usage of the vulnerability classification system
"""

from src.data_processor import VulnerabilityDataProcessor
from src.model import VulnerabilityPredictor
import json


def demo_data_processing():
    """Demonstrate data processing"""
    print("\n" + "="*60)
    print("DEMO: Data Processing")
    print("="*60)
    
    processor = VulnerabilityDataProcessor('config.json')
    
    # Load sample data
    print("\nLoading sample data...")
    data_dict = processor.prepare_data('data/sample_vulnerabilities.csv')
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(data_dict['train']) + len(data_dict['val']) + len(data_dict['test'])}")
    print(f"  Training samples: {len(data_dict['train'])}")
    print(f"  Validation samples: {len(data_dict['val'])}")
    print(f"  Test samples: {len(data_dict['test'])}")
    
    print(f"\nSample descriptions from training set:")
    for i, row in data_dict['train'].head(3).iterrows():
        print(f"\n  {i+1}. {row['description'][:80]}...")
        print(f"     Type: {row['vulnerability_type']}, Severity: {row['severity']}")


def demo_prediction():
    """Demonstrate prediction (requires trained model)"""
    print("\n" + "="*60)
    print("DEMO: Prediction")
    print("="*60)
    
    # Sample vulnerability descriptions
    samples = [
        "SQL injection vulnerability allows attackers to execute arbitrary SQL commands through user input fields",
        "Cross-site scripting vulnerability in comment section allows execution of malicious JavaScript code",
        "Buffer overflow in network protocol parser can lead to remote code execution",
        "Weak password policy allows easy brute force attacks"
    ]
    
    try:
        print("\nLoading model...")
        predictor = VulnerabilityPredictor('models/best_model.pt', 'config.json')
        
        print("\nMaking predictions...\n")
        for i, desc in enumerate(samples, 1):
            result = predictor.predict(desc)
            print(f"{i}. Description: {desc[:70]}...")
            print(f"   Type: {result['type_label']} (confidence: {result['type_confidence']:.2%})")
            print(f"   Severity: {result['severity_label']} (confidence: {result['severity_confidence']:.2%})")
            print()
            
    except FileNotFoundError:
        print("\nModel not found. Please train the model first using:")
        print("  cd src && python train.py")
        print("\nNote: Training requires a properly formatted dataset.")


def show_config():
    """Display current configuration"""
    print("\n" + "="*60)
    print("DEMO: Configuration")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("\nModel Configuration:")
    for key, value in config['model'].items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in config['training'].items():
        print(f"  {key}: {value}")
    
    print("\nVulnerability Types:")
    for i, vtype in enumerate(config['vulnerability_types'], 1):
        print(f"  {i}. {vtype}")
    
    print("\nSeverity Levels:")
    for i, severity in enumerate(config['severity_levels'], 1):
        print(f"  {i}. {severity}")


def main():
    """Run all demos"""
    print("\n")
    print("*" * 60)
    print(" BERT-based Vulnerability Classification System - DEMO")
    print("*" * 60)
    
    # Show configuration
    show_config()
    
    # Demo data processing
    demo_data_processing()
    
    # Demo prediction
    demo_prediction()
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Prepare your vulnerability dataset in CSV format")
    print("  2. Train the model: cd src && python train.py")
    print("  3. Evaluate: cd src && python evaluate.py")
    print("  4. Make predictions: cd src && python predict.py --description 'your text'")
    print()


if __name__ == '__main__':
    main()
