# Contributing to Financial Explainable Summarization

Thank you for your interest in contributing! This project aims to advance explainable AI for financial text summarization.

---

## 🎯 Ways to Contribute

### 1. Report Bugs
- Use GitHub Issues
- Include: Python version, OS, error logs
- Provide minimal reproducible example

### 2. Suggest Features
- Open a Feature Request issue
- Describe use case and benefits
- Discuss implementation approach

### 3. Improve Documentation
- Fix typos or unclear sections
- Add examples or tutorials
- Translate documentation

### 4. Submit Code
- Fix bugs
- Implement features
- Add tests
- Optimize performance

---

## 🚀 Getting Started

### Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/financial-explainable-summarization-using-external-knowledge.git
cd financial-explainable-summarization-using-external-knowledge

# Add upstream
git remote add upstream https://github.com/sumeerriaz2020/financial-explainable-summarization-using-external-knowledge.git
```

### Set Up Development Environment

```bash
# Create environment
conda env create -f environment.yml
conda activate fin-explainable

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### Create a Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## 💻 Development Workflow

### 1. Make Changes

- Follow existing code style
- Add docstrings to functions/classes
- Update type hints
- Keep commits atomic and focused

### 2. Write Tests

```bash
# Add tests in tests/
# Test file: test_your_feature.py

# Run tests
pytest test/test_your_feature.py -v

# Check coverage
pytest test/ --cov=. --cov-report=html
```

**Requirement:** New code must have ≥80% test coverage.

### 3. Format Code

```bash
# Auto-format with black
black algorithms/ models/ utils/

# Check with flake8
flake8 algorithms/ models/ utils/

# Type checking
mypy algorithms/ models/ utils/
```

### 4. Update Documentation

- Add docstrings (Google style)
- Update README if needed
- Add to CHANGELOG.md

### 5. Commit

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "feat: add multi-stakeholder feedback mechanism

- Implement RL-based feedback collection
- Add stakeholder preference learning
- Update MESA framework with new profiles
- Add tests for feedback mechanism

Closes #123"
```

**Commit Message Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Add/update tests
- `chore`: Maintenance tasks

---

## 📝 Code Style Guide

### Python Style (PEP 8)

```python
"""
Module docstring: Brief description.

Longer description if needed.
"""

import standard_library
import third_party_library
from local_module import LocalClass


class ExampleClass:
    """
    Class docstring.
    
    Attributes:
        attribute_name (type): Description
    """
    
    def __init__(self, param: str):
        """Initialize class.
        
        Args:
            param (str): Parameter description
        """
        self.attribute = param
    
    def method(self, arg: int) -> float:
        """
        Method docstring.
        
        Args:
            arg (int): Argument description
            
        Returns:
            float: Return value description
            
        Raises:
            ValueError: When invalid input
        """
        if arg < 0:
            raise ValueError("arg must be positive")
        
        return float(arg * 2)
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Tuple

def process_documents(
    documents: List[str],
    max_length: int = 128
) -> Tuple[List[str], Dict[str, float]]:
    """Process documents."""
    summaries = []
    metrics = {}
    return summaries, metrics
```

### Docstrings (Google Style)

```python
def complex_function(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, any]:
    """
    Brief one-line description.
    
    More detailed description if needed. Explain what the
    function does, not how it does it.
    
    Args:
        param1 (str): Description of param1
        param2 (Optional[int], optional): Description of param2.
            Defaults to None.
    
    Returns:
        Dict[str, any]: Description of return value with structure
    
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not int
    
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['key'])
        'value'
    """
    pass
```

---

## 🧪 Testing Guidelines

### Test Structure

```python
import unittest
from your_module import YourClass

class TestYourClass(unittest.TestCase):
    """Test YourClass functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.instance = YourClass()
    
    def test_feature_with_valid_input(self):
        """Test feature with valid input."""
        result = self.instance.method("valid")
        self.assertEqual(result, expected)
    
    def test_feature_with_invalid_input(self):
        """Test feature raises error with invalid input."""
        with self.assertRaises(ValueError):
            self.instance.method("invalid")
```

### Test Coverage Requirements

- **Minimum:** 80% overall coverage
- **New Features:** 90% coverage required
- **Critical Components:** 95% coverage

### Running Tests

```bash
# Run all tests
pytest test/ -v

# Run specific test file
pytest test/test_your_feature.py -v

# Run with coverage
pytest test/ --cov=. --cov-report=html

# Run specific test
pytest test/test_file.py::TestClass::test_method -v
```

---

## 📋 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Coverage ≥80%
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts

### Submit PR

1. **Push to your fork:**
```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request on GitHub:**
- Title: Clear, concise description
- Description: Explain changes, motivation, testing
- Link related issues

3. **PR Template:**
```markdown
## Description
Brief description of changes.

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
- [ ] Added tests
- [ ] All tests pass
- [ ] Coverage ≥80%

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

Closes #issue_number
```

### Review Process

- Maintainers will review within 3-5 days
- Address review comments
- Keep discussion professional and constructive
- Update PR based on feedback

### Merge

Once approved:
- Maintainer will merge
- Delete your branch after merge
- Pull latest main before next contribution

---

## 🐛 Bug Reports

### Before Reporting

- Search existing issues
- Check if bug exists in latest version
- Try to isolate the problem

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Load model with '...'
2. Call method '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]

**Error logs**
```python
# Paste error traceback
```

**Additional context**
Any other relevant information.
```

---

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution**
How would you like this to work?

**Describe alternatives**
Other solutions you've considered.

**Use case**
Real-world scenario where this helps.

**Additional context**
Mockups, examples, references.
```

---

## 📚 Documentation Contributions

### Documentation Style

- Clear, concise language
- Use examples liberally
- Include code snippets
- Add visual aids (diagrams, tables)

### Building Docs Locally

```bash
# Install docs dependencies
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs/
make html

# View
open _build/html/index.html
```

---

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in paper (for major contributions)

---

## 📞 Communication

### Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, ideas
- **Email:** sumeer33885@iqraisb.edu.pk (maintainers)

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- No harassment or discrimination

---

## 📖 Resources

### Learn About the Project

- [README.md](README.md) - Project overview
- [docs/](docs/) - Detailed documentation
- [Paper](paper.pdf) - Research paper

### Learn Technologies

- [PyTorch](https://pytorch.org/docs/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [FIBO Ontology](https://spec.edmcouncil.org/fibo/)

---

## ❓ Questions?

If you have questions:
1. Check documentation
2. Search existing issues
3. Ask in GitHub Discussions
4. Email maintainers

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

**Happy Contributing! 🚀**
