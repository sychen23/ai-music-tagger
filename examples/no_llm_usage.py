"""
Example: Using the pipeline without LLM (rule-based fallback)
"""
from music_tagger import MusicTaggerPipeline

# Initialize pipeline (no API key needed)
pipeline = MusicTaggerPipeline()

# Process without LLM - uses rule-based tagging
metadata = pipeline.process_file(
    "path/to/your/song.mp3",
    transcribe=False,  # Skip transcription to save time
    use_llm=False      # Use rule-based fallback
)

print(f"Title: {metadata.title}")
print(f"Mood: {', '.join(metadata.mood)}")
print(f"Audio Features:")
print(f"  Tempo: {metadata.audio_features.tempo} BPM")
print(f"  Key: {metadata.audio_features.key}")
print(f"  Energy: {metadata.audio_features.energy:.2f}")
print(f"  Valence: {metadata.audio_features.valence:.2f}")
