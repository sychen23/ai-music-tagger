"""
AI Music Tagger - An open-source AI music metadata tagging pipeline.

This package provides tools for automatically labeling songs using:
- Audio-to-text models for transcription
- Audio feature extraction for musical characteristics
- LLM-based reasoning for intelligent tagging
"""

__version__ = "0.1.0"

from .pipeline import MusicTaggerPipeline
from .models import MusicMetadata

__all__ = ["MusicTaggerPipeline", "MusicMetadata"]
