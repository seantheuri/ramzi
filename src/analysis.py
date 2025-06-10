import librosa
import numpy as np
import json
import argparse
import os
from openai import OpenAI


def estimate_key(y, sr):
    """
    Estimates the key of a track using librosa's chromagram and a key correlation template.
    Returns the key in standard notation (e.g., "C# minor") and Camelot notation (e.g., "12A").
    """
    # Get chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Define templates for all major and minor keys
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    
    # Generate all key profiles
    major_profiles = [np.roll(major_template, i) for i in range(12)]
    minor_profiles = [np.roll(minor_template, i) for i in range(12)]
    
    # Aggregate chroma features over time
    chroma_agg = np.sum(chroma, axis=1)
    chroma_agg = chroma_agg / np.sum(chroma_agg) # Normalize

    # Calculate correlations
    major_corrs = [np.corrcoef(chroma_agg, prof)[0, 1] for prof in major_profiles]
    minor_corrs = [np.corrcoef(chroma_agg, prof)[0, 1] for prof in minor_profiles]
    
    # Find the best match
    best_major_idx = np.argmax(major_corrs)
    best_minor_idx = np.argmax(minor_corrs)
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    if major_corrs[best_major_idx] > minor_corrs[best_minor_idx]:
        key_standard = f"{notes[best_major_idx]} Major"
        key_mode = "Major"
        key_idx = best_major_idx
    else:
        key_standard = f"{notes[best_minor_idx]} Minor"
        key_mode = "Minor"
        key_idx = best_minor_idx

    # Camelot Wheel Mapping (Key: Standard, Value: [Camelot Major, Camelot Minor])
    camelot_map = {
        'B': ['1B', '10A'], 'F#': ['2B', '11A'], 'D-flat': ['2B', '11A'],
        'D-flat': ['3B', '12A'], 'C#': ['3B', '12A'], 'A-flat': ['4B', '1A'], 'G#': ['4B', '1A'],
        'E-flat': ['5B', '2A'], 'D#': ['5B', '2A'], 'B-flat': ['6B', '3A'], 'A#': ['6B', '3A'],
        'F': ['7B', '4A'], 'C': ['8B', '5A'], 'G': ['9B', '6A'],
        'D': ['10B', '7A'], 'A': ['11B', '8A'], 'E': ['12B', '9A']
    }
    
    note_name = notes[key_idx]
    if key_mode == "Major":
        key_camelot = camelot_map.get(note_name, [None, None])[0]
    else:
        key_camelot = camelot_map.get(note_name, [None, None])[1]

    return key_standard, key_camelot


def label_segments(segment_times, y, sr):
    """
    Applies heuristics to label segments based on their relative loudness (RMS).
    Labels: intro, outro, chorus, verse, breakdown.
    """
    if len(segment_times) <= 1:
        # Not enough segments to analyze, label the whole thing as 'main'
        return [{"label": "main", "start": 0.0, "end": librosa.get_duration(y=y, sr=sr)}]

    labeled_segments = []
    segment_rms = []
    
    for i in range(len(segment_times) - 1):
        start_sample = librosa.time_to_samples(segment_times[i], sr=sr)
        end_sample = librosa.time_to_samples(segment_times[i+1], sr=sr)
        segment_y = y[start_sample:end_sample]
        rms = np.mean(librosa.feature.rms(y=segment_y))
        segment_rms.append(rms)

    max_rms = max(segment_rms) if segment_rms else 0

    for i, rms in enumerate(segment_rms):
        start_time = segment_times[i]
        end_time = segment_times[i+1]
        label = "verse" # Default label

        # Heuristics for labeling
        if i == 0 and rms < max_rms * 0.7:
            label = "intro"
        elif i == len(segment_rms) - 1 and rms < max_rms * 0.7:
            label = "outro"
        elif rms >= max_rms * 0.9: # Loudest parts are choruses/drops
            label = "chorus"
        elif rms < max_rms * 0.4: # Quietest parts are breakdowns
            label = "breakdown"
        
        labeled_segments.append({"label": label, "start": start_time, "end": end_time})

    return labeled_segments

def analyze_song(file_path, include_lyrics=True):
    """
    Main analysis function. Takes a file path and returns a dictionary of audio features.
    """
    print(f"Analyzing: {file_path}...")
    
    # 1. Load audio file
    try:
        y, sr = librosa.load(file_path, sr=None) # Load with original sample rate
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

    # 2. BPM and Beat Grid
    print("  - Calculating BPM and Beat Grid...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # 3. Energy (using RMS as a proxy)
    print("  - Calculating Energy...")
    rms = librosa.feature.rms(y=y)
    # Normalize energy to a 0-1 scale for easier use
    avg_energy = np.mean(rms)
    normalized_energy = (avg_energy - np.min(rms)) / (np.max(rms) - np.min(rms)) if np.max(rms) > np.min(rms) else 0.0

    # 4. Key Estimation
    print("  - Estimating Key...")
    key_standard, key_camelot = estimate_key(y, sr)
    
    # 5. Structural Analysis
    print("  - Performing Structural Analysis...")
    # Use chroma features for segmentation, as they are good at capturing harmonic structure
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Use an agglomerative clustering approach to find segment boundaries
    # The number of segments is a parameter you can tune
    bounds = librosa.segment.agglomerative(chroma, k=10) # k is target number of segments
    segment_times = librosa.frames_to_time(bounds, sr=sr)
    # Add start and end of track to segment times
    segment_times = np.concatenate(([0.0], segment_times, [librosa.get_duration(y=y, sr=sr)]))
    
    labeled_structure = label_segments(segment_times, y, sr)

    # 6. Transcription with OpenAI Whisper API
    lyrics_timed = []
    if include_lyrics:
        print("  - Transcribing Lyrics with OpenAI API (this may take a while)...")
        try:
            client = OpenAI()
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            
            # Extract timestamped words from the response
            if hasattr(transcription, 'words') and transcription.words:
                lyrics_timed = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    }
                    for word in transcription.words
                ]
                print(f"  - Transcribed {len(lyrics_timed)} words")
            else:
                print("  - No word-level timestamps available")
                
        except Exception as e:
            print(f"  - OpenAI transcription failed: {e}")
            print("  - Make sure you have OPENAI_API_KEY set in your environment")
            lyrics_timed = []
    else:
        print("  - Skipping lyrics transcription")

    # 7. Assemble final data object
    output_data = {
        "file_path": file_path,
        "bpm": str(round(tempo[0], 2)),
        "key_standard": key_standard,
        "key_camelot": key_camelot,
        "energy_normalized": str(round(normalized_energy, 3)),
        "beat_grid_seconds": beat_times.tolist(),
        "structural_analysis": labeled_structure,
        "lyrics_timed": lyrics_timed
    }
    
    print("Analysis complete.")
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a song file to extract DJing information.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (e.g., mp3, wav).")
    parser.add_argument("-o", "--output_file", type=str, help="Path to save the output JSON file. Defaults to input filename with .json extension.")
    parser.add_argument("--no-lyrics", action="store_true", help="Skip lyrics transcription (saves API costs).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
    else:
        # Analyze song with or without lyrics
        include_lyrics = not args.no_lyrics
        analysis_result = analyze_song(args.input_file, include_lyrics=include_lyrics)
        
        if analysis_result:
            output_path = args.output_file
            if not output_path:
                base_name = os.path.splitext(args.input_file)[0].split("/")[-1]
                output_path = f"analysis/{base_name}_analysis.json"
            
            with open(output_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            print(f"\nSuccessfully saved analysis to: {output_path}")
            
