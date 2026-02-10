"""
Tests for LLM tagger with mocked API calls.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from music_tagger.llm_tagger import LLMTagger
from openai import RateLimitError, APIError


@pytest.fixture
def mock_openai_client():
    """Fixture for mocked OpenAI client."""
    with patch('music_tagger.llm_tagger.OpenAI') as mock_client:
        yield mock_client


def test_generate_tags_with_api_success(mock_openai_client):
    """Test successful API call for tag generation."""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '''
    {
        "title": "Summer Vibes",
        "artist": "Unknown",
        "genre": ["Pop", "Electronic"],
        "mood": ["Happy", "Energetic"],
        "instruments": ["Synth", "Drums"],
        "tags": ["Upbeat"],
        "description": "An energetic track"
    }
    '''

    # Configure mock client
    mock_instance = mock_openai_client.return_value
    mock_instance.chat.completions.create.return_value = mock_response

    # Create tagger with mocked client
    tagger = LLMTagger(api_key="fake-key")

    # Test tag generation
    audio_features = {"tempo": 120, "energy": 0.8, "valence": 0.9}
    result = tagger.generate_tags(
        lyrics="Test lyrics",
        audio_features=audio_features,
        file_name="test.mp3"
    )

    # Assertions
    assert result["title"] == "Summer Vibes"
    assert "Pop" in result["genre"]
    assert "Happy" in result["mood"]
    assert mock_instance.chat.completions.create.called


def test_generate_tags_with_rate_limit_retry(mock_openai_client):
    """Test rate limit handling with retry."""
    # Setup mock to fail first time, succeed second time
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '''
    {
        "title": "Test Song",
        "artist": "Unknown",
        "genre": ["Pop"],
        "mood": ["Happy"],
        "instruments": [],
        "tags": [],
        "description": "Test"
    }
    '''

    mock_instance = mock_openai_client.return_value
    mock_instance.chat.completions.create.side_effect = [
        RateLimitError("Rate limit exceeded", response=Mock(status_code=429), body={}),
        mock_response
    ]

    # Create tagger with fast retry
    tagger = LLMTagger(api_key="fake-key", rate_limit_delay=0.1, max_retries=3)

    # Test tag generation
    result = tagger.generate_tags(
        audio_features={"tempo": 120},
        file_name="test.mp3"
    )

    # Should succeed after retry
    assert result["title"] == "Test Song"
    assert mock_instance.chat.completions.create.call_count == 2


def test_generate_tags_fallback_after_max_retries(mock_openai_client):
    """Test fallback tagging after max retries exceeded."""
    # Setup mock to always fail
    mock_instance = mock_openai_client.return_value
    mock_instance.chat.completions.create.side_effect = RateLimitError(
        "Rate limit exceeded",
        response=Mock(status_code=429),
        body={}
    )

    # Create tagger with fast retry
    tagger = LLMTagger(api_key="fake-key", rate_limit_delay=0.1, max_retries=2)

    # Test tag generation
    audio_features = {"tempo": 150, "energy": 0.9, "valence": 0.8}
    result = tagger.generate_tags(
        audio_features=audio_features,
        file_name="test.mp3"
    )

    # Should use fallback tagging
    assert result["title"] == "Test"
    assert "Fast" in result["mood"]  # From high tempo
    assert mock_instance.chat.completions.create.call_count == 2


def test_generate_tags_api_error_retry(mock_openai_client):
    """Test API error handling with retry."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '''
    {"title": "Test", "artist": "Unknown", "genre": [], "mood": [],
     "instruments": [], "tags": [], "description": ""}
    '''

    mock_instance = mock_openai_client.return_value
    mock_instance.chat.completions.create.side_effect = [
        APIError("API Error", request=Mock(), body={}),
        mock_response
    ]

    tagger = LLMTagger(api_key="fake-key", rate_limit_delay=0.1, max_retries=3)

    result = tagger.generate_tags(
        audio_features={"tempo": 120},
        file_name="test.mp3"
    )

    assert result["title"] == "Test"
    assert mock_instance.chat.completions.create.call_count == 2


def test_rate_limit_delay():
    """Test that rate limiting applies delay between calls."""
    import time

    tagger = LLMTagger(api_key=None, rate_limit_delay=0.2)

    start_time = time.time()
    tagger._apply_rate_limit()
    first_call_time = time.time()

    tagger._apply_rate_limit()
    second_call_time = time.time()

    # Second call should be delayed
    delay = second_call_time - first_call_time
    assert delay >= 0.2, f"Expected delay >= 0.2s, got {delay}s"


def test_generate_tags_without_api_key():
    """Test that tagger falls back when no API key is provided."""
    tagger = LLMTagger(api_key=None)

    audio_features = {"tempo": 120, "energy": 0.5, "valence": 0.5}
    result = tagger.generate_tags(
        audio_features=audio_features,
        file_name="test.mp3"
    )

    # Should use fallback tagging
    assert result["title"] == "Test"
    assert isinstance(result["mood"], list)
    assert len(result["mood"]) > 0


def test_context_building():
    """Test context string building for LLM prompt."""
    tagger = LLMTagger(api_key=None)

    context = tagger._build_context(
        lyrics="Test lyrics here",
        audio_features={"tempo": 120, "key": "C Major"},
        file_name="song.mp3"
    )

    assert "song.mp3" in context
    assert "Test lyrics here" in context
    assert "tempo" in context
    assert "120" in context


def test_initialization_with_custom_params():
    """Test initialization with custom rate limiting parameters."""
    tagger = LLMTagger(
        api_key=None,
        model="gpt-4",
        rate_limit_delay=2.0,
        max_retries=5
    )

    assert tagger.model == "gpt-4"
    assert tagger.rate_limit_delay == 2.0
    assert tagger.max_retries == 5
