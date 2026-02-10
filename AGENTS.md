# AI Music Tagger - Agent Guidelines

This document provides guidelines for AI agents working on the AI Music Tagger codebase.

## Project Overview

AI-powered music metadata tagging pipeline using:
- Audio feature extraction (librosa)
- Speech-to-text transcription (OpenAI Whisper)
- LLM-based metadata generation (OpenAI GPT)

## Build & Development Commands

```bash
# Install dependencies (Poetry - preferred)
poetry install

# Or using pip
pip install -r requirements.txt

# Run tests
pytest

# Run single test
pytest tests/test_models.py::test_audio_features_creation -v

# Run tests with coverage
pytest --cov=music_tagger --cov-report=term-missing

# Format code
black music_tagger/ tests/

# Lint code
flake8 music_tagger/ tests/

# Run CLI
poetry run music-tagger tag song.mp3
python -m music_tagger.cli tag song.mp3
```

## Code Style Guidelines

### Python Version & Formatting
- **Python**: 3.8+ required
- **Formatter**: Black (default settings)
- **Linter**: flake8
- **Line length**: 88 characters (Black default)

### Imports
```python
# Standard library imports first
import json
import os
from typing import List, Optional, Dict, Any

# Third-party imports
import click
import librosa
from pydantic import BaseModel, Field

# Local/package imports (relative within package)
from .models import MusicMetadata
from .pipeline import MusicTaggerPipeline
```

### Type Hints
- Use type hints for all function parameters and return values
- Use `Optional[X]` instead of `Union[X, None]`
- Use `List[X]`, `Dict[K, V]` from typing module

```python
def process_file(self, file_path: str, transcribe: bool = True) -> MusicMetadata:
    """Process a single audio file."""
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Error Handling
- Use specific exceptions (`FileNotFoundError`, `ValueError`)
- Handle API errors with retry logic and fallbacks
- Log errors with descriptive messages

```python
try:
    result = api_call()
except RateLimitError as e:
    print(f"Rate limit hit. Retrying...")
    time.sleep(delay)
except Exception as e:
    print(f"Error: {e}")
    return fallback_result
```

### Docstrings
Use Google-style docstrings:

```python
def method(self, param: str) -> ReturnType:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        FileNotFoundError: When file doesn't exist
    """
```

### Data Models (Pydantic)
```python
class AudioFeatures(BaseModel):
    """Audio features extracted from music file."""
    tempo: Optional[float] = Field(None, description="Tempo in BPM")
    key: Optional[str] = Field(None, description="Musical key")
```

### Testing
- Tests in `tests/` directory
- Use pytest fixtures for setup
- Mock external API calls in unit tests
- Test function naming: `test_<function_name>_<scenario>()`

```python
def test_music_metadata_defaults():
    """Test MusicMetadata with minimal fields."""
    metadata = MusicMetadata(file_path="/path/to/song.mp3")
    assert metadata.title is None
```

## Project Structure

```
music_tagger/
├── __init__.py          # Package init with lazy imports
├── cli.py               # CLI using Click
├── pipeline.py          # Main orchestration
├── models.py            # Pydantic data models
├── audio_processor.py   # Feature extraction
├── transcriber.py       # Whisper transcription
└── llm_tagger.py        # LLM metadata generation

tests/
├── test_models.py
├── test_pipeline.py
└── test_llm_tagger.py
```

## Dependencies

Core dependencies (see pyproject.toml):
- transformers (Hugging Face)
- torch
- librosa (audio analysis)
- openai
- pydantic (v2)
- click (CLI)

Dev dependencies:
- pytest
- pytest-mock
- black
- flake8

## Environment Setup

Create `.env` file:
```env
OPENAI_API_KEY=your-api-key-here
```

## Key Design Patterns

1. **Lazy Loading**: Models loaded on first use, not at import
2. **Fallback Mechanisms**: Rule-based tagging when LLM unavailable
3. **Rate Limiting**: Built-in delays between API calls
4. **Progress Indicators**: Use tqdm for batch operations
