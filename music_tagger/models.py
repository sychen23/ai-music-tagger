"""
Data models for music metadata.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AudioFeatures(BaseModel):
    """Audio features extracted from the music file."""
    tempo: Optional[float] = Field(None, description="Tempo in BPM")
    key: Optional[str] = Field(None, description="Musical key")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    energy: Optional[float] = Field(None, description="Energy level (0-1)")
    valence: Optional[float] = Field(None, description="Musical positiveness (0-1)")


class MusicMetadata(BaseModel):
    """Complete metadata for a music file."""
    file_path: str = Field(..., description="Path to the audio file")
    title: Optional[str] = Field(None, description="Song title")
    artist: Optional[str] = Field(None, description="Artist name")
    genre: Optional[List[str]] = Field(default_factory=list, description="Musical genres")
    mood: Optional[List[str]] = Field(default_factory=list, description="Mood tags")
    instruments: Optional[List[str]] = Field(default_factory=list, description="Detected instruments")
    lyrics: Optional[str] = Field(None, description="Transcribed lyrics")
    tags: Optional[List[str]] = Field(default_factory=list, description="Additional tags")
    audio_features: Optional[AudioFeatures] = Field(None, description="Audio features")
    description: Optional[str] = Field(None, description="AI-generated description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/song.mp3",
                "title": "Summer Vibes",
                "artist": "Unknown",
                "genre": ["Pop", "Electronic"],
                "mood": ["Happy", "Energetic"],
                "instruments": ["Piano", "Drums", "Synth"],
                "tags": ["Upbeat", "Dance"],
                "audio_features": {
                    "tempo": 120.0,
                    "key": "C Major",
                    "duration": 180.5,
                    "energy": 0.8,
                    "valence": 0.9
                }
            }
        }
