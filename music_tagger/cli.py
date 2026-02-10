"""
Command-line interface for the music tagger.
"""
import click
import json
import csv
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from .pipeline import MusicTaggerPipeline
from .models import MusicMetadata

# Load environment variables
load_dotenv()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """AI Music Tagger - Automatically generate metadata for music files."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path (JSON or CSV)')
@click.option('--no-transcribe', is_flag=True, help='Skip audio transcription')
@click.option('--no-llm', is_flag=True, help='Skip LLM tagging (use rule-based fallback)')
@click.option('--whisper-model', default='openai/whisper-base', help='Whisper model name')
@click.option('--llm-model', default='gpt-3.5-turbo', help='OpenAI model name')
def tag(input_path, output, no_transcribe, no_llm, whisper_model, llm_model):
    """
    Tag audio file(s) with AI-generated metadata.
    
    INPUT_PATH can be a single audio file or a directory of audio files.
    """
    # Initialize pipeline
    pipeline = MusicTaggerPipeline(
        whisper_model=whisper_model,
        llm_model=llm_model
    )
    
    # Determine input files
    input_path = Path(input_path)
    if input_path.is_file():
        file_paths = [str(input_path)]
    else:
        # Find all audio files in directory
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        file_paths = [
            str(f) for f in input_path.rglob('*')
            if f.suffix.lower() in audio_extensions
        ]
        
        if not file_paths:
            click.echo(f"No audio files found in {input_path}", err=True)
            return
        
        click.echo(f"Found {len(file_paths)} audio file(s)")
    
    # Process files
    transcribe = not no_transcribe
    use_llm = not no_llm
    
    if use_llm and not os.getenv('OPENAI_API_KEY'):
        click.echo("Warning: OPENAI_API_KEY not found. Using rule-based tagging fallback.", err=True)
        use_llm = False
    
    results = pipeline.process_batch(
        file_paths,
        transcribe=transcribe,
        use_llm=use_llm
    )
    
    # Output results
    if output:
        save_results(results, output)
        click.echo(f"\n✓ Results saved to: {output}")
    else:
        # Print to console
        for metadata in results:
            print_metadata(metadata)


@main.command()
@click.argument('audio_file', type=click.Path(exists=True))
def info(audio_file):
    """Display detailed information about an audio file."""
    pipeline = MusicTaggerPipeline()
    
    click.echo(f"\nAnalyzing: {audio_file}")
    click.echo("=" * 60)
    
    metadata = pipeline.process_file(audio_file, transcribe=True, use_llm=True)
    print_metadata(metadata, detailed=True)


def print_metadata(metadata: MusicMetadata, detailed: bool = False):
    """Print metadata in a readable format."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"File: {metadata.file_path}")
    click.echo(f"Title: {metadata.title or 'N/A'}")
    click.echo(f"Artist: {metadata.artist or 'N/A'}")
    
    if metadata.genre:
        click.echo(f"Genre: {', '.join(metadata.genre)}")
    
    if metadata.mood:
        click.echo(f"Mood: {', '.join(metadata.mood)}")
    
    if metadata.instruments:
        click.echo(f"Instruments: {', '.join(metadata.instruments)}")
    
    if metadata.tags:
        click.echo(f"Tags: {', '.join(metadata.tags)}")
    
    if metadata.audio_features:
        click.echo(f"\nAudio Features:")
        click.echo(f"  Tempo: {metadata.audio_features.tempo:.1f} BPM")
        click.echo(f"  Key: {metadata.audio_features.key}")
        click.echo(f"  Duration: {metadata.audio_features.duration:.1f}s")
        click.echo(f"  Energy: {metadata.audio_features.energy:.2f}")
        click.echo(f"  Valence: {metadata.audio_features.valence:.2f}")
    
    if detailed and metadata.description:
        click.echo(f"\nDescription:")
        click.echo(f"  {metadata.description}")
    
    if detailed and metadata.lyrics:
        click.echo(f"\nLyrics/Vocals:")
        click.echo(f"  {metadata.lyrics[:200]}..." if len(metadata.lyrics) > 200 else f"  {metadata.lyrics}")


def save_results(results: List[MusicMetadata], output_path: str):
    """Save results to file (JSON or CSV)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.json':
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            data = [metadata.model_dump() for metadata in results]
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif output_path.suffix.lower() == '.csv':
        # Save as CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if not results:
                return
            
            fieldnames = [
                'file_path', 'title', 'artist', 'genre', 'mood',
                'instruments', 'tags', 'tempo', 'key', 'duration',
                'energy', 'valence', 'description'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metadata in results:
                row = {
                    'file_path': metadata.file_path,
                    'title': metadata.title or '',
                    'artist': metadata.artist or '',
                    'genre': '|'.join(metadata.genre) if metadata.genre else '',
                    'mood': '|'.join(metadata.mood) if metadata.mood else '',
                    'instruments': '|'.join(metadata.instruments) if metadata.instruments else '',
                    'tags': '|'.join(metadata.tags) if metadata.tags else '',
                    'tempo': metadata.audio_features.tempo if metadata.audio_features else '',
                    'key': metadata.audio_features.key if metadata.audio_features else '',
                    'duration': metadata.audio_features.duration if metadata.audio_features else '',
                    'energy': metadata.audio_features.energy if metadata.audio_features else '',
                    'valence': metadata.audio_features.valence if metadata.audio_features else '',
                    'description': metadata.description or ''
                }
                writer.writerow(row)
    else:
        # Default to JSON
        save_results(results, str(output_path.with_suffix('.json')))


if __name__ == '__main__':
    main()
