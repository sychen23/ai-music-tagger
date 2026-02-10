#!/usr/bin/env python3
"""
Demo script showing the AI Music Tagger pipeline functionality.

This script demonstrates the pipeline without requiring actual audio files
by mocking the components and showing expected inputs/outputs.
"""
import json
from music_tagger.models import MusicMetadata, AudioFeatures


def demo_metadata_creation():
    """Demonstrate creating music metadata."""
    print("=" * 70)
    print("AI MUSIC TAGGER - DEMO")
    print("=" * 70)
    print()
    
    # Example 1: Create metadata with all fields
    print("Example 1: Complete metadata for an energetic pop song")
    print("-" * 70)
    
    audio_features = AudioFeatures(
        tempo=128.0,
        key="C Major",
        duration=195.5,
        energy=0.82,
        valence=0.91
    )
    
    metadata = MusicMetadata(
        file_path="/path/to/summer_vibes.mp3",
        title="Summer Vibes",
        artist="Electronic Dreams",
        genre=["Pop", "Electronic", "Dance"],
        mood=["Happy", "Energetic", "Uplifting", "Positive"],
        instruments=["Synth", "Drums", "Bass", "Piano"],
        lyrics="Feel the sunshine, dancing all night long...",
        tags=["Upbeat", "Commercial", "Radio-friendly", "Party", "Festival"],
        audio_features=audio_features,
        description="An energetic, uplifting pop track with electronic elements. Features prominent synths and a driving beat at 128 BPM, perfect for commercial use, fitness content, or upbeat promotional videos."
    )
    
    # Display formatted output
    print(f"Title: {metadata.title}")
    print(f"Artist: {metadata.artist}")
    print(f"Genre: {', '.join(metadata.genre)}")
    print(f"Mood: {', '.join(metadata.mood)}")
    print(f"Instruments: {', '.join(metadata.instruments)}")
    print(f"Tags: {', '.join(metadata.tags)}")
    print(f"\nAudio Features:")
    print(f"  Tempo: {metadata.audio_features.tempo} BPM")
    print(f"  Key: {metadata.audio_features.key}")
    print(f"  Duration: {metadata.audio_features.duration}s")
    print(f"  Energy: {metadata.audio_features.energy:.2f}")
    print(f"  Valence: {metadata.audio_features.valence:.2f}")
    print(f"\nDescription:")
    print(f"  {metadata.description}")
    print()
    
    # Export to JSON
    json_output = metadata.model_dump()
    print("JSON Output (formatted):")
    print(json.dumps(json_output, indent=2))
    print()
    
    # Example 2: Minimal metadata (rule-based fallback)
    print("\n" + "=" * 70)
    print("Example 2: Minimal metadata from rule-based analysis")
    print("-" * 70)
    
    audio_features2 = AudioFeatures(
        tempo=72.0,
        key="A Minor",
        duration=240.0,
        energy=0.35,
        valence=0.28
    )
    
    metadata2 = MusicMetadata(
        file_path="/path/to/ambient_track.wav",
        title="Ambient Track",
        artist="Unknown",
        genre=["Ambient", "Electronic"],
        mood=["Calm", "Melancholic", "Atmospheric"],
        instruments=["Synth Pad", "Piano"],
        tags=["Slow", "Relaxed", "Background"],
        audio_features=audio_features2,
        description="A slow, calm ambient track with a tempo of 72 BPM. Perfect for meditation, relaxation, or background music."
    )
    
    print(f"Title: {metadata2.title}")
    print(f"Mood: {', '.join(metadata2.mood)}")
    print(f"Tempo: {metadata2.audio_features.tempo} BPM (Slow)")
    print(f"Energy: {metadata2.audio_features.energy:.2f} (Low)")
    print(f"Description: {metadata2.description}")
    print()
    
    # Example 3: Use case for sync licensing
    print("\n" + "=" * 70)
    print("Example 3: Metadata optimized for sync licensing platform")
    print("-" * 70)
    
    metadata3 = MusicMetadata(
        file_path="/path/to/corporate_background.mp3",
        title="Corporate Success",
        artist="Production Music Library",
        genre=["Corporate", "Pop", "Inspirational"],
        mood=["Confident", "Uplifting", "Professional", "Motivational"],
        instruments=["Piano", "Strings", "Acoustic Guitar", "Light Drums"],
        tags=[
            "Business", "Corporate", "Commercial", "Advertisement",
            "Presentation", "Success", "Achievement", "Growth"
        ],
        audio_features=AudioFeatures(
            tempo=110.0,
            key="G Major",
            duration=180.0,
            energy=0.65,
            valence=0.82
        ),
        description="A confident and uplifting corporate track featuring piano, strings, and acoustic guitar. Perfect for business presentations, corporate videos, commercials, and motivational content. The 110 BPM tempo creates a sense of steady progress and achievement."
    )
    
    print(f"Title: {metadata3.title}")
    print(f"Use Case Tags: {', '.join(metadata3.tags[:5])}")
    print(f"Mood Profile: {', '.join(metadata3.mood)}")
    print(f"Best For: Corporate videos, presentations, commercials")
    print()
    
    print("=" * 70)
    print("PIPELINE WORKFLOW")
    print("=" * 70)
    print("""
The complete AI Music Tagger pipeline:

1. AUDIO LOADING
   - Supports: MP3, WAV, FLAC, M4A, OGG
   - Loads audio and resamples to 16kHz

2. AUDIO FEATURE EXTRACTION (librosa)
   - Tempo/BPM detection
   - Musical key estimation
   - Energy level (RMS)
   - Valence (spectral features)
   - Duration

3. AUDIO TRANSCRIPTION (Whisper AI)
   - Transcribe vocals and lyrics
   - Identify speech/singing in audio
   - Extract text content

4. LLM-BASED TAGGING (GPT)
   Input:
   - Audio features (tempo, key, energy, etc.)
   - Transcribed lyrics/vocals
   - File name
   
   Output:
   - Genre tags
   - Mood descriptors
   - Instrument list
   - Additional tags
   - Human-readable description

5. METADATA OUTPUT
   - JSON format for APIs
   - CSV format for catalogs
   - Structured data models

FALLBACK MODE (No API key required):
   - Rule-based tempo classification
   - Energy/valence-based mood tags
   - Basic metadata extraction
    """)
    
    print("=" * 70)
    print("Installation & Usage")
    print("=" * 70)
    print("""
# Install
pip install -r requirements.txt

# Set API key (optional, for LLM features)
# Option 1: Environment variable
export OPENAI_API_KEY="your-key"

# Option 2: .env file (recommended)
cp .env.example .env
# Edit .env and add your API key

# Use CLI
music-tagger tag song.mp3
music-tagger tag music_folder/ --output catalog.json

# Use Python API
from music_tagger import MusicTaggerPipeline

pipeline = MusicTaggerPipeline()
metadata = pipeline.process_file("song.mp3")
print(metadata.genre, metadata.mood)
    """)


if __name__ == "__main__":
    demo_metadata_creation()
