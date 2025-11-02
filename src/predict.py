"""
Prediction script for vulnerability classification
"""

import argparse
import json
from model import VulnerabilityPredictor


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(
        description='Predict vulnerability type and severity'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.pt',
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--description',
        type=str,
        required=True,
        help='Vulnerability description to classify'
    )
    
    args = parser.parse_args()
    
    # Load predictor
    print("Loading model...")
    predictor = VulnerabilityPredictor(args.model, args.config)
    
    # Make prediction
    print("\nPredicting...")
    result = predictor.predict(args.description)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"\nInput Description:")
    print(f"  {args.description}")
    print(f"\nVulnerability Type:")
    print(f"  {result['type_label']}")
    print(f"  Confidence: {result['type_confidence']:.2%}")
    print(f"\nSeverity:")
    print(f"  {result['severity_label']}")
    print(f"  Confidence: {result['severity_confidence']:.2%}")
    print("="*50)


if __name__ == '__main__':
    main()
