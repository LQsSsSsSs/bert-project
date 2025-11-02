# Contributing to BERT Vulnerability Classification System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Use cases and benefits
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add or update tests as needed
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Add comments for complex logic

### Testing

- Ensure all existing tests pass
- Add tests for new functionality
- Test edge cases
- Verify code works with different Python versions (3.8+)

### Documentation

- Update README.md if adding new features
- Update ARCHITECTURE.md for architectural changes
- Add examples to EXAMPLES.md for new use cases
- Keep docstrings up to date

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/LQsSsSsSs/-
cd -
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install in development mode:
```bash
pip install -e .
```

## Project Structure

```
.
├── src/              # Source code
├── data/             # Sample data
├── models/           # Trained models
├── tests/            # Test files (when added)
├── docs/             # Additional documentation
└── examples/         # Example scripts
```

## Coding Guidelines

### Python Code

```python
# Good
def calculate_accuracy(predictions, labels):
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)

# Avoid
def calc_acc(p, l):
    return sum(p[i] == l[i] for i in range(len(p))) / len(p)
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

Examples:
```
Add support for custom BERT models
Fix memory leak in data processor
Update README with installation instructions
```

### Branch Naming

- `feature/` for new features
- `bugfix/` for bug fixes
- `docs/` for documentation updates
- `refactor/` for code refactoring

Examples:
```
feature/add-ensemble-models
bugfix/fix-tokenization-error
docs/update-api-reference
refactor/simplify-training-loop
```

## Areas for Contribution

### High Priority

- [ ] Add comprehensive unit tests
- [ ] Support for more vulnerability types
- [ ] Model performance optimization
- [ ] API server implementation
- [ ] Docker containerization

### Medium Priority

- [ ] Attention visualization
- [ ] Model interpretability (LIME/SHAP)
- [ ] Support for multilingual descriptions
- [ ] Active learning pipeline
- [ ] Model quantization

### Low Priority

- [ ] Web UI for predictions
- [ ] Prometheus metrics export
- [ ] Integration with vulnerability databases
- [ ] Automated hyperparameter tuning
- [ ] Model ensemble support

## Questions?

Feel free to open an issue with your question, or contact the maintainers directly.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

Examples of behavior that contributes to a positive environment:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

Examples of unacceptable behavior:
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by opening an issue or contacting the project maintainers.

Thank you for contributing!
