# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Rate limiting and retry logic for OpenAI API calls in batch processing
- Progress bars using tqdm for batch processing operations
- Configurable language support for audio transcription
- Comprehensive test suite with mocked API calls
- Model caching documentation and configuration options
- Type hints throughout the codebase

### Changed
- Improved version pinning strategy in requirements.txt for more predictable builds
- Enhanced README with dependency size warnings and performance disclaimers

### Fixed
- Copyright year updated to 2026
- Author information in pyproject.toml
- Minor documentation issues and type hints

## [0.1.0] - 2026-02-10

### Added
- Initial release of AI Music Tagger
- Audio feature extraction using librosa (tempo, key, energy, valence)
- Whisper-based audio transcription for lyrics and vocals
- LLM-based intelligent metadata tagging using GPT models
- Rule-based fallback mode for operation without API keys
- CLI tool with `tag` and `info` commands
- Python API for programmatic use
- Batch processing support for multiple files
- JSON and CSV export formats
- Comprehensive documentation (README, QUICKSTART, CONTRIBUTING)
- Usage examples and demo script
- Unit tests covering core functionality
- MIT License

### Features
- **Audio Processor**: Extracts musical features (tempo, key, duration, energy, valence)
- **Transcriber**: Uses OpenAI Whisper models for speech-to-text
- **LLM Tagger**: Generates genre, mood, instrument tags and descriptions
- **Pipeline**: Orchestrates complete tagging workflow
- **CLI**: Command-line interface with multiple options
- **Lazy Loading**: Optimized imports to reduce startup time
- **GPU Support**: CUDA acceleration for Whisper inference

### Use Cases
- Sync licensing platforms
- Music distributors and catalogs
- Content management systems
- Music library organization

### Dependencies
- Python 3.8+
- PyTorch (GPU support optional)
- Transformers (Hugging Face)
- librosa for audio analysis
- OpenAI API (optional)
- click for CLI
- pydantic for data validation

[unreleased]: https://github.com/sychen23/ai-music-tagger/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sychen23/ai-music-tagger/releases/tag/v0.1.0
