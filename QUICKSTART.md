# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/sychen23/ai-music-tagger.git
cd ai-music-tagger

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your-api-key-here
```

## Basic Usage

### Command Line

```bash
# Tag a single file
music-tagger tag my_song.mp3

# Tag all files in a directory
music-tagger tag /path/to/music/

# Save results to JSON
music-tagger tag song.mp3 --output results.json

# Save results to CSV for catalogs
music-tagger tag music_folder/ --output catalog.csv

# Get detailed information about a file
music-tagger info song.mp3

# Skip transcription for faster processing
music-tagger tag song.mp3 --no-transcribe

# Use without OpenAI API (rule-based fallback)
music-tagger tag song.mp3 --no-llm
```

### Python API

```python
from music_tagger import MusicTaggerPipeline

# Initialize pipeline
pipeline = MusicTaggerPipeline()

# Process single file
metadata = pipeline.process_file("song.mp3")

# Access metadata
print(f"Title: {metadata.title}")
print(f"Genre: {metadata.genre}")
print(f"Mood: {metadata.mood}")
print(f"Tempo: {metadata.audio_features.tempo} BPM")
print(f"Description: {metadata.description}")

# Process multiple files
files = ["song1.mp3", "song2.mp3", "song3.mp3"]
results = pipeline.process_batch(files)

# Export to JSON
import json
with open("output.json", "w") as f:
    data = [m.model_dump() for m in results]
    json.dump(data, f, indent=2)
```

## Advanced Usage

### Custom Models

```python
# Use larger Whisper model for better transcription
pipeline = MusicTaggerPipeline(
    whisper_model="openai/whisper-large-v3",
    llm_model="gpt-4",
    device="cuda"  # Use GPU
)
```

### Without LLM (Offline Mode)

```python
# Process without requiring API key
metadata = pipeline.process_file(
    "song.mp3",
    transcribe=False,  # Skip transcription
    use_llm=False      # Use rule-based tagging
)
```

### Batch Processing with Custom Settings

```python
import glob

# Find all audio files
audio_files = glob.glob("music/**/*.mp3", recursive=True)

# Process with custom settings
pipeline = MusicTaggerPipeline(
    whisper_model="openai/whisper-tiny",  # Faster
    llm_model="gpt-3.5-turbo"
)

results = pipeline.process_batch(
    audio_files,
    transcribe=True,
    use_llm=True
)

# Generate CSV report
import csv
with open("catalog.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "title", "artist", "genre", "mood", "tempo", "key", "description"
    ])
    writer.writeheader()
    for m in results:
        writer.writerow({
            "title": m.title,
            "artist": m.artist,
            "genre": "|".join(m.genre),
            "mood": "|".join(m.mood),
            "tempo": m.audio_features.tempo,
            "key": m.audio_features.key,
            "description": m.description
        })
```

## Output Format

### JSON Structure

```json
{
  "file_path": "/path/to/song.mp3",
  "title": "Song Title",
  "artist": "Artist Name",
  "genre": ["Pop", "Electronic"],
  "mood": ["Happy", "Energetic"],
  "instruments": ["Piano", "Drums"],
  "lyrics": "Transcribed lyrics...",
  "tags": ["Upbeat", "Commercial"],
  "audio_features": {
    "tempo": 120.0,
    "key": "C Major",
    "duration": 180.0,
    "energy": 0.8,
    "valence": 0.9
  },
  "description": "A description of the track..."
}
```

## Common Use Cases

### Sync Licensing Platform

```python
# Tag music library for licensing
import glob

pipeline = MusicTaggerPipeline()
music_files = (
    glob.glob("library/**/*.mp3", recursive=True) +
    glob.glob("library/**/*.wav", recursive=True)
)

results = pipeline.process_batch(music_files)

# Export for database import
for metadata in results:
    # Store in database
    db.add_track({
        "file": metadata.file_path,
        "searchable_tags": metadata.genre + metadata.mood + metadata.tags,
        "description": metadata.description,
        "tempo": metadata.audio_features.tempo,
        "key": metadata.audio_features.key
    })
```

### Music Distribution Catalog

```python
# Enrich catalog metadata
distributor_files = get_catalog_files()

for file in distributor_files:
    metadata = pipeline.process_file(file)
    
    # Update catalog entry
    update_catalog_entry(
        file_id=file.id,
        genre=metadata.genre,
        mood=metadata.mood,
        description=metadata.description
    )
```

## Performance Tips

1. **Use smaller Whisper models for speed**: `whisper-tiny` or `whisper-base`
2. **Skip transcription if not needed**: `--no-transcribe` flag
3. **Use GPU for faster processing**: Set `device="cuda"`
4. **Batch processing is more efficient**: Process multiple files at once
5. **Cache results**: Save processed metadata to avoid reprocessing

## Troubleshooting

### Memory Issues

```python
# Use smaller models
pipeline = MusicTaggerPipeline(
    whisper_model="openai/whisper-tiny",
    device="cpu"
)
```

### Slow Processing

```bash
# Skip transcription
music-tagger tag music/ --no-transcribe --output fast_results.json
```

### API Rate Limits

```python
# Add delay between LLM calls
import time

for file in audio_files:
    metadata = pipeline.process_file(file)
    time.sleep(1)  # 1 second delay
```

## Support

- Documentation: See [README.md](README.md)
- Issues: [GitHub Issues](https://github.com/sychen23/ai-music-tagger/issues)
- Examples: See `examples/` directory
