"""
Main pipeline for music tagging.
"""
from typing import Optional, List
import os
from tqdm import tqdm

from .models import MusicMetadata, AudioFeatures
from .audio_processor import AudioProcessor
from .transcriber import AudioTranscriber
from .llm_tagger import LLMTagger


class MusicTaggerPipeline:
    """
    Complete pipeline for AI music metadata tagging.
    
    This pipeline:
    1. Loads and analyzes audio files
    2. Extracts audio features (tempo, key, etc.)
    3. Transcribes vocals/lyrics using Whisper
    4. Generates metadata tags using LLM reasoning
    """
    
    def __init__(self,
                 whisper_model: str = "openai/whisper-base",
                 llm_model: str = "gpt-3.5-turbo",
                 openai_api_key: Optional[str] = None,
                 device: Optional[str] = None,
                 language: Optional[str] = None,
                 rate_limit_delay: float = 1.0,
                 max_retries: int = 3):
        """
        Initialize the music tagging pipeline.

        Args:
            whisper_model: Whisper model name for transcription
            llm_model: OpenAI model for metadata generation
            openai_api_key: OpenAI API key (optional, can use env var)
            device: Device for model inference ('cuda', 'cpu', or None for auto)
            language: Language code for transcription (e.g., 'en', 'es', 'fr').
                     If None, Whisper will auto-detect the language.
            rate_limit_delay: Delay in seconds between API calls (default: 1.0)
            max_retries: Maximum number of retries for failed API calls (default: 3)
        """
        self.audio_processor = AudioProcessor()
        self.transcriber = AudioTranscriber(
            model_name=whisper_model,
            device=device,
            language=language
        )
        self.llm_tagger = LLMTagger(
            api_key=openai_api_key,
            model=llm_model,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries
        )
    
    def process_file(self, 
                    file_path: str,
                    transcribe: bool = True,
                    use_llm: bool = True) -> MusicMetadata:
        """
        Process a single audio file and generate metadata.
        
        Args:
            file_path: Path to the audio file
            transcribe: Whether to transcribe lyrics (can be slow)
            use_llm: Whether to use LLM for tagging (requires API key)
            
        Returns:
            MusicMetadata object with all extracted information
        """
        print(f"\nProcessing: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Extract audio features
        print("Extracting audio features...")
        features_dict = self.audio_processor.extract_features(file_path)
        audio_features = AudioFeatures(**features_dict)
        
        # Transcribe audio
        lyrics = None
        if transcribe:
            print("Transcribing audio...")
            transcription_result = self.transcriber.transcribe(file_path)
            if transcription_result.get("success"):
                lyrics = transcription_result.get("text", "")
                print(f"Transcribed: {lyrics[:100]}..." if len(lyrics) > 100 else f"Transcribed: {lyrics}")
            else:
                print("Transcription failed or no speech detected")
        
        # Generate metadata using LLM
        print("Generating metadata tags...")
        file_name = os.path.basename(file_path)
        
        if use_llm:
            llm_tags = self.llm_tagger.generate_tags(
                lyrics=lyrics,
                audio_features=features_dict,
                file_name=file_name
            )
        else:
            llm_tags = self.llm_tagger._fallback_tagging(
                lyrics=lyrics,
                audio_features=features_dict,
                file_name=file_name
            )
        
        # Combine all metadata
        metadata = MusicMetadata(
            file_path=file_path,
            title=llm_tags.get("title"),
            artist=llm_tags.get("artist"),
            genre=llm_tags.get("genre", []),
            mood=llm_tags.get("mood", []),
            instruments=llm_tags.get("instruments", []),
            lyrics=lyrics,
            tags=llm_tags.get("tags", []),
            audio_features=audio_features,
            description=llm_tags.get("description")
        )
        
        print("✓ Processing complete")
        return metadata
    
    def process_batch(self,
                     file_paths: List[str],
                     transcribe: bool = True,
                     use_llm: bool = True,
                     show_progress: bool = True) -> List[MusicMetadata]:
        """
        Process multiple audio files.

        Args:
            file_paths: List of paths to audio files
            transcribe: Whether to transcribe lyrics
            use_llm: Whether to use LLM for tagging
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of MusicMetadata objects
        """
        results = []

        # Create progress bar iterator
        iterator = tqdm(file_paths, desc="Processing files", unit="file", disable=not show_progress)

        for file_path in iterator:
            try:
                # Update progress bar with current file
                if show_progress:
                    iterator.set_postfix_str(os.path.basename(file_path)[:30])

                metadata = self.process_file(file_path, transcribe=transcribe, use_llm=use_llm)
                results.append(metadata)
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error processing {file_path}: {e}")
                else:
                    print(f"Error processing {file_path}: {e}")
                continue

        return results
