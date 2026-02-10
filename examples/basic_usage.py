"""
Example: Basic usage of the music tagging pipeline
"""
from music_tagger import MusicTaggerPipeline

# Initialize the pipeline
pipeline = MusicTaggerPipeline()

# Process a single audio file
metadata = pipeline.process_file(
    "path/to/your/song.mp3",
    transcribe=True,  # Transcribe vocals/lyrics
    use_llm=True      # Use LLM for intelligent tagging
)

# Print results
print(f"Title: {metadata.title}")
print(f"Artist: {metadata.artist}")
print(f"Genre: {', '.join(metadata.genre)}")
print(f"Mood: {', '.join(metadata.mood)}")
print(f"Tempo: {metadata.audio_features.tempo} BPM")
print(f"Key: {metadata.audio_features.key}")
print(f"\nDescription: {metadata.description}")

# Save to JSON
import json
with open("output.json", "w") as f:
    json.dump(metadata.model_dump(), f, indent=2)
