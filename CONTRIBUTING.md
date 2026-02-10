# AI Music Tagger Development

## Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black flake8
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=music_tagger --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## Code Style

```bash
# Format code with black
black music_tagger/

# Lint with flake8
flake8 music_tagger/
```

## Project Structure

```
ai-music-tagger/
├── music_tagger/          # Main package
│   ├── __init__.py
│   ├── models.py          # Data models
│   ├── audio_processor.py # Audio feature extraction
│   ├── transcriber.py     # Whisper transcription
│   ├── llm_tagger.py      # LLM-based tagging
│   ├── pipeline.py        # Main pipeline
│   └── cli.py             # Command-line interface
├── tests/                 # Unit tests
├── examples/              # Usage examples
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Adding New Features

1. Create a new branch
2. Implement the feature
3. Add tests
4. Update documentation
5. Submit a pull request

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Create a git tag
4. Push to GitHub
5. GitHub Actions will build and publish
