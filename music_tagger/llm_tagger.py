"""
LLM-based music tagging using OpenAI API.
"""
from openai import OpenAI
from typing import Dict, List, Optional
import json
import os


class LLMTagger:
    """Generate music metadata using LLM reasoning."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM tagger.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_tags(self, 
                     lyrics: Optional[str] = None,
                     audio_features: Optional[Dict] = None,
                     file_name: Optional[str] = None) -> Dict:
        """
        Generate music metadata tags using LLM.
        
        Args:
            lyrics: Transcribed lyrics or vocals
            audio_features: Dictionary of audio features
            file_name: Name of the audio file
            
        Returns:
            Dictionary with generated metadata
        """
        if not self.client:
            # Return simple rule-based tags if no API key
            return self._fallback_tagging(lyrics, audio_features, file_name)
        
        # Build context for LLM
        context = self._build_context(lyrics, audio_features, file_name)
        
        # Create prompt
        prompt = self._create_prompt(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a music industry expert specializing in metadata tagging for sync licensing and catalog systems. Provide accurate, detailed tags for music files."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"LLM tagging error: {e}")
            return self._fallback_tagging(lyrics, audio_features, file_name)
    
    def _build_context(self, 
                      lyrics: Optional[str],
                      audio_features: Optional[Dict],
                      file_name: Optional[str]) -> str:
        """Build context string from available information."""
        parts = []
        
        if file_name:
            parts.append(f"File name: {file_name}")
        
        if audio_features:
            parts.append(f"Audio features: {json.dumps(audio_features, indent=2)}")
        
        if lyrics:
            parts.append(f"Lyrics/Vocals: {lyrics}")
        
        return "\n\n".join(parts)
    
    def _create_prompt(self, context: str) -> str:
        """Create the LLM prompt."""
        return f"""Analyze this music file and generate comprehensive metadata tags suitable for sync licensing platforms, distributors, and catalog systems.

{context}

Provide a JSON response with the following fields:
- title: Suggested title (if not obvious from file name, use descriptive title)
- artist: Artist name if identifiable, otherwise "Unknown"
- genre: Array of 1-3 genre tags (e.g., ["Pop", "Electronic"])
- mood: Array of 2-5 mood descriptors (e.g., ["Happy", "Energetic", "Uplifting"])
- instruments: Array of detected/likely instruments (e.g., ["Piano", "Drums", "Guitar"])
- tags: Array of additional descriptive tags (e.g., ["Upbeat", "Commercial", "Radio-friendly"])
- description: A 2-3 sentence description of the music suitable for catalog listings

Be specific and accurate. Focus on tags that would be useful for music supervisors, sync licensing, and content discovery."""
    
    def _fallback_tagging(self,
                         lyrics: Optional[str],
                         audio_features: Optional[Dict],
                         file_name: Optional[str]) -> Dict:
        """Generate basic tags without LLM (rule-based fallback)."""
        result = {
            "title": self._extract_title(file_name) if file_name else "Unknown",
            "artist": "Unknown",
            "genre": [],
            "mood": [],
            "instruments": [],
            "tags": [],
            "description": ""
        }
        
        # Extract some basic info from audio features
        if audio_features:
            tempo = audio_features.get("tempo", 0)
            energy = audio_features.get("energy", 0)
            valence = audio_features.get("valence", 0)
            
            # Tempo-based tags
            if tempo > 140:
                result["mood"].append("Fast")
                result["tags"].append("High-energy")
            elif tempo < 80:
                result["mood"].append("Slow")
                result["tags"].append("Relaxed")
            else:
                result["mood"].append("Moderate")
            
            # Energy-based tags
            if energy > 0.7:
                result["mood"].append("Energetic")
            elif energy < 0.3:
                result["mood"].append("Calm")
            
            # Valence-based tags
            if valence > 0.7:
                result["mood"].append("Happy")
                result["mood"].append("Positive")
            elif valence < 0.3:
                result["mood"].append("Melancholic")
            
            result["description"] = f"A {result['mood'][0].lower() if result['mood'] else 'musical'} track"
            if tempo:
                result["description"] += f" with a tempo of {int(tempo)} BPM"
        
        # Add lyrics info if available
        if lyrics and len(lyrics.strip()) > 10:
            result["tags"].append("Vocal")
            result["instruments"].append("Vocals")
        
        return result
    
    def _extract_title(self, file_name: str) -> str:
        """Extract a readable title from file name."""
        # Remove extension and clean up
        name = os.path.splitext(os.path.basename(file_name))[0]
        # Replace underscores and hyphens with spaces
        name = name.replace("_", " ").replace("-", " ")
        # Capitalize words
        return name.title()
