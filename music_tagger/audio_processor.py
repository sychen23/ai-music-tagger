"""
Audio processing utilities for loading and analyzing audio files.
"""
import librosa
import numpy as np
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AudioProcessor:
    """Process audio files and extract features."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def extract_features(self, file_path: str) -> dict:
        """
        Extract audio features from a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary of audio features
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Extract key (using chroma features)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        key = self._estimate_key(chroma)
        
        # Duration
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=audio)
        energy = float(np.mean(rms))
        
        # Valence estimation (using spectral features)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        valence = float(np.mean(spectral_centroid) / (sr / 2))  # Normalize
        
        return {
            "tempo": float(tempo) if isinstance(tempo, (np.ndarray, np.number)) else tempo,
            "key": key,
            "duration": float(duration),
            "energy": min(energy * 10, 1.0),  # Normalize to 0-1
            "valence": min(valence, 1.0)
        }
    
    def _estimate_key(self, chroma: np.ndarray) -> str:
        """
        Estimate musical key from chroma features.
        
        Args:
            chroma: Chroma feature matrix
            
        Returns:
            Estimated key as string
        """
        # Average chroma over time
        chroma_mean = np.mean(chroma, axis=1)
        
        # Key names
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Find the most prominent note
        key_idx = int(np.argmax(chroma_mean))
        
        # Simple major/minor detection based on chroma pattern
        # This is a simplified approach
        major_pattern = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_pattern = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        # Roll patterns to match detected root
        major_rolled = np.roll(major_pattern, key_idx)
        minor_rolled = np.roll(minor_pattern, key_idx)
        
        # Correlation with patterns
        major_corr = np.corrcoef(chroma_mean, major_rolled)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rolled)[0, 1]
        
        mode = "Major" if major_corr > minor_corr else "Minor"
        
        return f"{keys[key_idx]} {mode}"
