"""
Tests for the LLM tagger module.
"""
import pytest
from music_tagger.llm_tagger import LLMTagger


def test_llm_tagger_initialization():
    """Test LLMTagger initialization."""
    tagger = LLMTagger(api_key=None)
    
    assert tagger.model == "gpt-3.5-turbo"
    assert tagger.client is None


def test_llm_tagger_extract_title():
    """Test title extraction from filename."""
    tagger = LLMTagger(api_key=None)
    
    title = tagger._extract_title("my_awesome_song.mp3")
    assert title == "My Awesome Song"
    
    title = tagger._extract_title("test-track-2024.wav")
    assert title == "Test Track 2024"


def test_fallback_tagging_basic():
    """Test fallback tagging without LLM."""
    tagger = LLMTagger(api_key=None)
    
    audio_features = {
        "tempo": 120,
        "energy": 0.8,
        "valence": 0.9
    }
    
    result = tagger._fallback_tagging(
        lyrics=None,
        audio_features=audio_features,
        file_name="test_song.mp3"
    )
    
    assert result["title"] == "Test Song"
    assert result["artist"] == "Unknown"
    assert isinstance(result["genre"], list)
    assert isinstance(result["mood"], list)
    assert len(result["mood"]) > 0


def test_fallback_tagging_tempo():
    """Test fallback tagging tempo-based classification."""
    tagger = LLMTagger(api_key=None)
    
    # Fast tempo
    result_fast = tagger._fallback_tagging(
        lyrics=None,
        audio_features={"tempo": 150, "energy": 0.8, "valence": 0.5},
        file_name="fast.mp3"
    )
    assert "Fast" in result_fast["mood"]
    
    # Slow tempo
    result_slow = tagger._fallback_tagging(
        lyrics=None,
        audio_features={"tempo": 70, "energy": 0.3, "valence": 0.5},
        file_name="slow.mp3"
    )
    assert "Slow" in result_slow["mood"]


def test_fallback_tagging_with_lyrics():
    """Test fallback tagging with lyrics."""
    tagger = LLMTagger(api_key=None)
    
    result = tagger._fallback_tagging(
        lyrics="This is a song with lyrics",
        audio_features={"tempo": 120, "energy": 0.5, "valence": 0.5},
        file_name="song.mp3"
    )
    
    assert "Vocal" in result["tags"]
    assert "Vocals" in result["instruments"]


def test_build_context():
    """Test context building for LLM."""
    tagger = LLMTagger(api_key=None)
    
    context = tagger._build_context(
        lyrics="Test lyrics",
        audio_features={"tempo": 120},
        file_name="test.mp3"
    )
    
    assert "test.mp3" in context
    assert "Test lyrics" in context
    assert "tempo" in context
