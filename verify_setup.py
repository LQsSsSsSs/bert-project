#!/usr/bin/env python3
"""
Quick Start Guide for Vulnerability Classification System
Run this script to verify the installation and setup
"""

import sys
import os


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False


def check_files():
    """Check if all required files exist"""
    print("\nChecking project structure...")
    required_files = [
        'config.json',
        'requirements.txt',
        'README.md',
        'src/data_processor.py',
        'src/model.py',
        'src/train.py',
        'src/predict.py',
        'src/evaluate.py',
        'data/sample_vulnerabilities.csv'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check if dependencies are installed"""
    print("\nChecking dependencies...")
    required_modules = [
        'torch',
        'transformers',
        'sklearn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} (not installed)")
            missing.append(module)
    
    return len(missing) == 0, missing


def show_instructions():
    """Show quick start instructions"""
    print("\n" + "="*70)
    print("QUICK START INSTRUCTIONS")
    print("="*70)
    
    print("""
1. Install Dependencies:
   pip install -r requirements.txt

2. Verify Installation:
   python demo.py

3. Prepare Your Data:
   - Create a CSV file with columns: description, vulnerability_type, severity
   - Or use the sample data: data/sample_vulnerabilities.csv

4. Train the Model:
   cd src
   python train.py

5. Make Predictions:
   cd src
   python predict.py --description "Your vulnerability description here"

6. Evaluate Model:
   cd src
   python evaluate.py

For more details, see README.md
""")


def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("VULNERABILITY CLASSIFICATION SYSTEM - SETUP VERIFICATION")
    print("="*70 + "\n")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check files
    files_ok = check_files()
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if python_ok and files_ok and deps_ok:
        print("\n✓ All checks passed! System is ready to use.")
        print("\nRun 'python demo.py' to see a demonstration.")
    else:
        print("\n✗ Some checks failed. Please address the issues above.")
        
        if not deps_ok:
            print("\nTo install missing dependencies, run:")
            print("  pip install -r requirements.txt")
    
    # Show instructions
    show_instructions()


if __name__ == '__main__':
    main()
