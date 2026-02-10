"""
Tests for the pipeline module.
"""
import pytest
from music_tagger.pipeline import MusicTaggerPipeline
from music_tagger.models import MusicMetadata


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = MusicTaggerPipeline()
    
    assert pipeline.audio_processor is not None
    assert pipeline.transcriber is not None
    assert pipeline.llm_tagger is not None


def test_pipeline_custom_models():
    """Test pipeline with custom models."""
    pipeline = MusicTaggerPipeline(
        whisper_model="openai/whisper-tiny",
        llm_model="gpt-4"
    )
    
    assert pipeline.transcriber.model_name == "openai/whisper-tiny"
    assert pipeline.llm_tagger.model == "gpt-4"


def test_pipeline_file_not_found():
    """Test pipeline with non-existent file."""
    pipeline = MusicTaggerPipeline()
    
    with pytest.raises(FileNotFoundError):
        pipeline.process_file("nonexistent.mp3")
