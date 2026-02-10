# AI Music Tagger 🎵🤖

An open-source, AI-powered music metadata tagging pipeline that automatically labels songs using audio-to-text (speech/music) models + LLMs for reasoning and tagging logic, useful for sync licensing platforms, distributors, and catalog systems.

## Features

- 🎼 **Audio Feature Extraction**: Automatically detect tempo (BPM), musical key, energy, valence, and duration
- 🎤 **Lyrics Transcription**: Use Whisper AI models to transcribe vocals and lyrics from audio
- 🧠 **Intelligent Tagging**: Leverage LLMs (GPT) for sophisticated music metadata generation
- 🏷️ **Comprehensive Metadata**: Generate genre, mood, instruments, tags, and descriptions
- 📦 **Batch Processing**: Process entire music catalogs efficiently
- 💾 **Multiple Output Formats**: Export to JSON or CSV for easy integration
- 🔌 **API & CLI**: Use as a Python library or command-line tool
- 🚀 **Fallback Mode**: Works without API keys using rule-based tagging

## Installation

```bash
# Clone the repository
git clone https://github.com/sychen23/ai-music-tagger.git
cd ai-music-tagger

# Install dependencies
pip install -r requirements.txt

# Or using poetry
poetry install
```

## Quick Start

### Command-Line Usage

```bash
# Set your OpenAI API key (optional, but recommended)
export OPENAI_API_KEY="your-api-key-here"

# Tag a single audio file
music-tagger tag song.mp3

# Tag all files in a directory and save to JSON
music-tagger tag /path/to/music/folder --output results.json

# Get detailed info about a file
music-tagger info song.mp3

# Batch process without transcription (faster)
music-tagger tag /path/to/music --no-transcribe --output results.csv
```

### Python API Usage

```python
from music_tagger import MusicTaggerPipeline

# Initialize the pipeline
pipeline = MusicTaggerPipeline()

# Process a single file
metadata = pipeline.process_file("song.mp3")

# Access the generated metadata
print(f"Title: {metadata.title}")
print(f"Genre: {', '.join(metadata.genre)}")
print(f"Mood: {', '.join(metadata.mood)}")
print(f"Tempo: {metadata.audio_features.tempo} BPM")
print(f"Key: {metadata.audio_features.key}")

# Process multiple files
results = pipeline.process_batch(["song1.mp3", "song2.mp3"])
```

## How It Works

The AI Music Tagger pipeline consists of four main components:

1. **Audio Processor** (`audio_processor.py`)
   - Loads audio files (MP3, WAV, FLAC, etc.)
   - Extracts musical features using librosa
   - Detects tempo, key, energy, and valence

2. **Audio Transcriber** (`transcriber.py`)
   - Uses OpenAI's Whisper model for speech-to-text
   - Transcribes vocals and lyrics from audio
   - Supports multiple Whisper model sizes

3. **LLM Tagger** (`llm_tagger.py`)
   - Generates metadata using GPT models
   - Analyzes audio features and lyrics
   - Creates genre, mood, and instrument tags
   - Generates human-readable descriptions

4. **Pipeline Orchestrator** (`pipeline.py`)
   - Coordinates all components
   - Manages the complete tagging workflow
   - Handles batch processing

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key-here
```

### Custom Models

```python
# Use a different Whisper model
pipeline = MusicTaggerPipeline(
    whisper_model="openai/whisper-large-v3",  # More accurate but slower
    llm_model="gpt-4",  # Better reasoning
    device="cuda"  # Use GPU acceleration
)
```

## Output Format

The pipeline generates structured metadata in the following format:

```json
{
  "file_path": "/path/to/song.mp3",
  "title": "Summer Vibes",
  "artist": "Unknown",
  "genre": ["Pop", "Electronic"],
  "mood": ["Happy", "Energetic", "Uplifting"],
  "instruments": ["Synth", "Drums", "Bass"],
  "lyrics": "Transcribed vocals...",
  "tags": ["Upbeat", "Commercial"],
  "audio_features": {
    "tempo": 128.0,
    "key": "C Major",
    "duration": 195.5,
    "energy": 0.82,
    "valence": 0.91
  },
  "description": "An energetic pop track with electronic elements..."
}
```

## Use Cases

### Sync Licensing Platforms
- Automatically tag music libraries for search and discovery
- Generate descriptions for music supervisors
- Categorize tracks by mood and genre

### Music Distributors
- Enrich catalog metadata at scale
- Standardize tagging across large collections
- Improve searchability and recommendations

### Content Management Systems
- Auto-tag user-uploaded music
- Build searchable music databases
- Enable content filtering and discovery

## CLI Commands

### `music-tagger tag`

Tag audio files with metadata.

```bash
music-tagger tag INPUT_PATH [OPTIONS]

Options:
  -o, --output PATH       Output file (JSON or CSV)
  --no-transcribe        Skip audio transcription
  --no-llm               Skip LLM tagging
  --whisper-model TEXT   Whisper model name
  --llm-model TEXT       OpenAI model name
```

### `music-tagger info`

Display detailed information about an audio file.

```bash
music-tagger info AUDIO_FILE
```

## Examples

See the `examples/` directory for more usage examples:

- `basic_usage.py` - Simple single-file processing
- `batch_processing.py` - Process multiple files
- `no_llm_usage.py` - Use without API keys

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- librosa
- OpenAI API key (optional, for LLM features)

## Performance

- **Without transcription**: ~2-5 seconds per file
- **With transcription**: ~10-30 seconds per file (depends on length and model)
- **GPU acceleration**: Significantly faster transcription with CUDA

## Limitations

- Transcription accuracy depends on vocal clarity
- LLM tagging requires an API key (fallback available)
- Audio feature extraction is rule-based (not ML-based)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI Whisper for speech recognition
- Hugging Face Transformers
- librosa for audio analysis

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
