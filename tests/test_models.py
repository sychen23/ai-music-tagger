"""
Tests for the models module.
"""
import pytest
from music_tagger.models import MusicMetadata, AudioFeatures


def test_audio_features_creation():
    """Test creating AudioFeatures object."""
    features = AudioFeatures(
        tempo=120.0,
        key="C Major",
        duration=180.5,
        energy=0.8,
        valence=0.9
    )
    
    assert features.tempo == 120.0
    assert features.key == "C Major"
    assert features.duration == 180.5
    assert features.energy == 0.8
    assert features.valence == 0.9


def test_audio_features_optional():
    """Test AudioFeatures with optional fields."""
    features = AudioFeatures()
    
    assert features.tempo is None
    assert features.key is None
    assert features.duration is None


def test_music_metadata_creation():
    """Test creating MusicMetadata object."""
    audio_features = AudioFeatures(tempo=128.0, key="D Minor")
    
    metadata = MusicMetadata(
        file_path="/path/to/song.mp3",
        title="Test Song",
        artist="Test Artist",
        genre=["Pop", "Rock"],
        mood=["Happy", "Energetic"],
        instruments=["Guitar", "Drums"],
        tags=["Upbeat"],
        audio_features=audio_features,
        description="A test song"
    )
    
    assert metadata.file_path == "/path/to/song.mp3"
    assert metadata.title == "Test Song"
    assert metadata.artist == "Test Artist"
    assert "Pop" in metadata.genre
    assert "Happy" in metadata.mood
    assert metadata.audio_features.tempo == 128.0


def test_music_metadata_defaults():
    """Test MusicMetadata with minimal fields."""
    metadata = MusicMetadata(file_path="/path/to/song.mp3")
    
    assert metadata.file_path == "/path/to/song.mp3"
    assert metadata.title is None
    assert metadata.genre == []
    assert metadata.mood == []
    assert metadata.instruments == []
    assert metadata.tags == []


def test_music_metadata_to_dict():
    """Test converting MusicMetadata to dictionary."""
    metadata = MusicMetadata(
        file_path="/path/to/song.mp3",
        title="Test Song",
        genre=["Rock"]
    )
    
    data = metadata.model_dump()
    
    assert isinstance(data, dict)
    assert data["file_path"] == "/path/to/song.mp3"
    assert data["title"] == "Test Song"
    assert data["genre"] == ["Rock"]
