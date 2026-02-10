"""
AI Music Tagger - An open-source AI music metadata tagging pipeline.

This package provides tools for automatically labeling songs using:
- Audio-to-text models for transcription
- Audio feature extraction for musical characteristics
- LLM-based reasoning for intelligent tagging
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy dependencies on module import
def __getattr__(name):
    if name == "MusicTaggerPipeline":
        from .pipeline import MusicTaggerPipeline
        return MusicTaggerPipeline
    elif name == "MusicMetadata":
        from .models import MusicMetadata
        return MusicMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MusicTaggerPipeline", "MusicMetadata"]
