#!/usr/bin/env python3
"""
Example Usage: Phase 1 Analysis Pipeline
========================================

This script demonstrates how to use the music analysis pipeline
with the new structure where Spotify features are direct attributes.
"""

from pathlib import Path
from src.analysis import AudioAnalyzer, create_analyzer_from_env

def demonstrate_analysis():
    """Demonstrate the analysis pipeline with the new structure"""
    
    # Find an audio file to analyze
    audio_files = list(Path('.').glob('*.mp3')) + list(Path('.').glob('*.wav'))
    
    if not audio_files:
        print("No audio files found. Please add some .mp3 or .wav files to test.")
        return
    
    test_file = audio_files[0]
    print(f"Analyzing: {test_file}")
    
    # Create analyzer
    analyzer = create_analyzer_from_env()
    
    # Run analysis
    result = analyzer.analyze_track(
        file_path=test_file,
        include_spotify=True,
        include_structure_allin1=True,
        include_structure_msaf=False,
        include_lyrics=False,
        timeout_minutes=1.0
    )
    
    # Display results with new structure
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    
    print(f"📁 File: {result.file_path}")
    print(f"🎵 Title: {result.title or 'Unknown'}")
    print(f"🎤 Artist: {result.artist or 'Unknown'}")
    print(f"🎼 BPM: {result.bpm:.1f}" if result.bpm else "🎼 BPM: Not detected")
    print(f"🗝️  Key: {result.key_standard} ({result.key_camelot})")
    
    # Spotify features as direct attributes
    print(f"\n🎧 Spotify Audio Features:")
    print(f"  Energy: {result.energy:.3f}" if result.energy else "  Energy: Not available")
    print(f"  Danceability: {result.danceability:.3f}" if result.danceability else "  Danceability: Not available")
    print(f"  Valence: {result.valence:.3f}" if result.valence else "  Valence: Not available")
    print(f"  Speechiness: {result.speechiness:.3f}" if result.speechiness else "  Speechiness: Not available")
    print(f"  Acousticness: {result.acousticness:.3f}" if result.acousticness else "  Acousticness: Not available")
    print(f"  Instrumentalness: {result.instrumentalness:.3f}" if result.instrumentalness else "  Instrumentalness: Not available")
    print(f"  Liveness: {result.liveness:.3f}" if result.liveness else "  Liveness: Not available")
    print(f"  Loudness: {result.loudness:.1f} dB" if result.loudness else "  Loudness: Not available")
    
    # Beat grid
    if result.beat_grid_seconds:
        print(f"\n🥁 Beat Grid: {len(result.beat_grid_seconds)} beats")
        print(f"  First 5 beats: {[f'{b:.2f}s' for b in result.beat_grid_seconds[:5]]}")
    
    # Structure analysis
    if result.structural_analysis:
        print(f"\n🏗️ Structure ({len(result.structural_analysis)} segments):")
        for i, segment in enumerate(result.structural_analysis):
            duration = segment['end_time'] - segment['start_time']
            print(f"  {i+1:2d}. {segment['label']:10s} "
                  f"{segment['start_time']:6.1f}s - {segment['end_time']:6.1f}s "
                  f"({duration:5.1f}s)")
    
    # Save to JSON with new structure
    output_file = f"{test_file.stem}_analysis.json"
    result.save_json(output_file)
    print(f"\n💾 Analysis saved to: {output_file}")
    
    # Show how to access features programmatically
    print(f"\n🔧 Programmatic Access Examples:")
    print(f"  result.energy = {result.energy}")
    print(f"  result.danceability = {result.danceability}")
    print(f"  result.bpm = {result.bpm}")
    print(f"  result.key_standard = '{result.key_standard}'")
    print(f"  result.key_camelot = '{result.key_camelot}'")

def show_json_structure():
    """Show what the JSON output looks like with the new structure"""
    print("\n" + "="*50)
    print("EXPECTED JSON STRUCTURE")
    print("="*50)
    
    sample_json = '''{
  "file_path": "songs/my_song.mp3",
  "title": "Song Title",
  "artist": "Artist Name",
  "bpm": 124.98,
  "key_camelot": "8A",
  "key_standard": "G Minor",
  "beat_grid_seconds": [0.48, 0.96, 1.44, "..."],
  "energy": 0.78,
  "danceability": 0.85,
  "valence": 0.6,
  "speechiness": 0.04,
  "acousticness": 0.12,
  "instrumentalness": 0.001,
  "liveness": 0.15,
  "loudness": -6.2,
  "structural_analysis": [
    {
      "label": "intro",
      "start_beat": 0,
      "end_beat": 64,
      "start_time": 0.0,
      "end_time": 30.7
    },
    {
      "label": "main",
      "start_beat": 64,
      "end_beat": 448,
      "start_time": 30.7,
      "end_time": 215.0
    }
  ],
  "lyrics_timed": [
    {
      "word": "The",
      "start": 31.2,
      "end": 31.4
    },
    {
      "word": "lights",
      "start": 31.5,
      "end": 31.9
    }
  ]
}'''
    print(sample_json)

if __name__ == "__main__":
    print("Phase 1 Analysis Pipeline - Updated Structure Demo")
    print("="*55)
    
    try:
        demonstrate_analysis()
        show_json_structure()
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have audio files and dependencies installed.")
    
    print("\n" + "="*55)
    print("✅ Demo completed!")
    print("Note: Spotify features are now direct attributes for easier access.") 