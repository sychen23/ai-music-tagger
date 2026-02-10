"""
Example: Batch processing multiple audio files
"""
from music_tagger import MusicTaggerPipeline
import json
from pathlib import Path

# Initialize the pipeline
pipeline = MusicTaggerPipeline()

# Get all audio files in a directory
audio_dir = Path("path/to/audio/files")
audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

# Process all files
print(f"Processing {len(audio_files)} files...")
results = pipeline.process_batch(
    [str(f) for f in audio_files],
    transcribe=True,
    use_llm=True
)

# Save results to JSON
output_data = [metadata.model_dump() for metadata in results]
with open("batch_results.json", "w") as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Processed {len(results)} files")

# Create a summary
for metadata in results:
    print(f"\n{metadata.title}")
    print(f"  Genre: {', '.join(metadata.genre)}")
    print(f"  Mood: {', '.join(metadata.mood)}")
    print(f"  Tempo: {metadata.audio_features.tempo:.1f} BPM")
