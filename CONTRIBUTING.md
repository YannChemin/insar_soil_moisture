# Contributing to InSAR Soil Moisture

Thank you for considering contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yannchemin/insar_soil_moisture/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GDAL version)
   - Full error traceback

### Suggesting Features

1. Check existing issues and discussions
2. Create a feature request issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternatives you've considered

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Format code: `black src/ tests/`
6. Commit with clear messages
7. Push and create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/insar_soil_moisture.git
cd insar_soil_moisture

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints where appropriate
- Write docstrings for all public functions/classes

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

## Documentation

- Update README if adding new features
- Add docstrings to new functions
- Update examples if API changes

## Questions?

Feel free to open an issue for questions or discussions.
