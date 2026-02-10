"""
Audio transcription using Whisper model.
"""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class AudioTranscriber:
    """Transcribe audio using Whisper model."""
    
    def __init__(self, model_name: str = "openai/whisper-base", device: Optional[str] = None):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Hugging Face model name for Whisper
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize model lazily
        self._pipe = None
    
    def _init_model(self):
        """Initialize the Whisper model (lazy loading)."""
        if self._pipe is not None:
            return
        
        print(f"Loading Whisper model: {self.model_name} on {self.device}...")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(self.model_name)
        
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        self._init_model()
        
        try:
            result = self._pipe(
                audio_path,
                return_timestamps=False,
                generate_kwargs={"language": "english"}
            )
            
            return {
                "text": result["text"].strip(),
                "success": True
            }
        except Exception as e:
            print(f"Transcription error: {e}")
            return {
                "text": "",
                "success": False,
                "error": str(e)
            }
