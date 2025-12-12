# Contributing to DMTS

Thank you for your interest in contributing to DMTS! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include details**:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Error messages and logs

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines below
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

## Code Style Guidelines

### Python

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and reasonably sized

### Example:

```python
def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text from source to target language.

    Args:
        text: The text to translate
        src_lang: Source language code (NLLB format)
        tgt_lang: Target language code (NLLB format)

    Returns:
        Translated text string
    """
    if not text or not text.strip():
        return ""
    # ... implementation
```

### Shell Scripts

- Use `#!/bin/bash` shebang
- Add comments for complex operations
- Use meaningful variable names in UPPERCASE
- Quote variables properly: `"${VAR}"`

## Testing

Before submitting:

1. **Test your changes locally**
2. **Verify imports work**:
   ```bash
   python -c "from dmts_mk4 import *"
   ```
3. **Run the server** and verify functionality
4. **Test with different backends** if applicable (NLLB, Hunyuan, Hybrid)

## Pull Request Guidelines

### Title Format

Use a clear, descriptive title:
- `fix: correct translation manager import path`
- `feat: add support for new language codes`
- `docs: update installation instructions`
- `refactor: simplify diarization clustering logic`

### Description

Include:
- What the PR does
- Why the change is needed
- Any breaking changes
- Testing performed

### Example:

```markdown
## Description
Fix hallucination detection threshold not being applied correctly.

## Changes
- Updated `_verify_transcription()` to use configurable threshold
- Added validation for threshold range (0-1)

## Testing
- Tested with verification enabled/disabled
- Verified threshold affects detection sensitivity
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/DMTS.git
   cd DMTS
   ```

2. Create a development environment:
   ```bash
   conda create -n dmts-dev python=3.10
   conda activate dmts-dev
   pip install -r requirements.txt
   ```

3. Make your changes and test

## Questions?

If you have questions about contributing:

1. Check existing issues and discussions
2. Open a new issue with the `question` label
3. Reach out via GitHub discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DMTS!
