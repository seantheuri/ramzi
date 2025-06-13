import librosa
import numpy as np
import json
import argparse
import os
from openai import OpenAI
import scipy.signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


def estimate_key(y, sr):
    """
    Enhanced key estimation using chromagram correlation with confidence scoring.
    Returns key in standard notation, Camelot notation, and confidence score.
    """
    # Get chroma features with higher resolution
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512, n_fft=4096)
    
    # Define templates for all major and minor keys (Krumhansl-Schmuckler profiles)
    major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Generate all key profiles
    major_profiles = [np.roll(major_template, i) for i in range(12)]
    minor_profiles = [np.roll(minor_template, i) for i in range(12)]
    
    # Aggregate chroma features over time with weighting
    chroma_agg = np.mean(chroma, axis=1)
    chroma_agg = chroma_agg / np.sum(chroma_agg)  # Normalize

    # Calculate correlations with enhanced scoring
    major_corrs = [np.corrcoef(chroma_agg, prof)[0, 1] for prof in major_profiles]
    minor_corrs = [np.corrcoef(chroma_agg, prof)[0, 1] for prof in minor_profiles]
    
    # Handle NaN correlations
    major_corrs = [0 if np.isnan(corr) else corr for corr in major_corrs]
    minor_corrs = [0 if np.isnan(corr) else corr for corr in minor_corrs]
    
    # Find the best match
    best_major_idx = np.argmax(major_corrs)
    best_minor_idx = np.argmax(minor_corrs)
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    if major_corrs[best_major_idx] > minor_corrs[best_minor_idx]:
        key_standard = f"{notes[best_major_idx]} Major"
        key_mode = 1  # Major
        key_idx = best_major_idx
        key_confidence = major_corrs[best_major_idx]
    else:
        key_standard = f"{notes[best_minor_idx]} Minor"
        key_mode = 0  # Minor
        key_idx = best_minor_idx
        key_confidence = minor_corrs[best_minor_idx]

    # Camelot Wheel Mapping
    camelot_map = {
        ('C', 1): '8B', ('C', 0): '5A',
        ('C#', 1): '3B', ('C#', 0): '12A',
        ('D', 1): '10B', ('D', 0): '7A',
        ('D#', 1): '5B', ('D#', 0): '2A',
        ('E', 1): '12B', ('E', 0): '9A',
        ('F', 1): '7B', ('F', 0): '4A',
        ('F#', 1): '2B', ('F#', 0): '11A',
        ('G', 1): '9B', ('G', 0): '6A',
        ('G#', 1): '4B', ('G#', 0): '1A',
        ('A', 1): '11B', ('A', 0): '8A',
        ('A#', 1): '6B', ('A#', 0): '3A',
        ('B', 1): '1B', ('B', 0): '10A'
    }
    
    key_camelot = camelot_map.get((notes[key_idx], key_mode), 'Unknown')

    return key_standard, key_camelot, key_idx, key_mode, key_confidence


def analyze_tempo_and_beats(y, sr):
    """
    Enhanced tempo and beat analysis with confidence scoring.
    """
    # Get tempo and beats with different hop lengths for accuracy
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    
    # Calculate tempo confidence based on beat consistency
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
        tempo_confidence = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
        tempo_confidence = max(0.0, min(1.0, tempo_confidence))
    else:
        tempo_confidence = 0.0
    
    return float(tempo), beat_times, tempo_confidence


def analyze_time_signature(y, sr, tempo):
    """
    Estimate time signature based on beat patterns.
    """
    # Get onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    if len(onset_times) < 8:
        return 4, 0.5  # Default to 4/4 with low confidence
    
    # Calculate beat period
    beat_period = 60.0 / tempo
    
    # Analyze onset patterns to detect time signature
    onset_intervals = np.diff(onset_times)
    
    # Look for patterns that suggest different time signatures
    beat_positions = (onset_times % (beat_period * 4)) / beat_period
    
    # Test for different time signatures
    signatures = [3, 4, 5, 6, 7]
    confidences = []
    
    for sig in signatures:
        # Calculate how well onsets align with this time signature
        expected_positions = np.arange(0, sig)
        alignment_scores = []
        
        for pos in beat_positions:
            distances = np.abs(expected_positions - (pos % sig))
            min_distance = np.min(distances)
            alignment_scores.append(1.0 - min_distance)
        
        confidence = np.mean(alignment_scores) if alignment_scores else 0.0
        confidences.append(confidence)
    
    best_sig_idx = np.argmax(confidences)
    time_signature = signatures[best_sig_idx]
    time_signature_confidence = confidences[best_sig_idx]
    
    # Ensure reasonable confidence bounds
    time_signature_confidence = max(0.0, min(1.0, time_signature_confidence))
    
    return time_signature, time_signature_confidence


def analyze_loudness(y, sr):
    """
    Comprehensive loudness analysis including overall, max, and dynamic range.
    """
    # Overall loudness (RMS-based)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    loudness = librosa.amplitude_to_db(np.mean(rms))
    
    # Peak loudness
    peak_loudness = librosa.amplitude_to_db(np.max(np.abs(y)))
    
    # Loudness range (dynamic range)
    loudness_range = peak_loudness - loudness
    
    return float(loudness), float(peak_loudness), float(loudness_range)


def analyze_spectral_features(y, sr):
    """
    Extract spectral features for detailed audio analysis.
    """
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Statistical measures
    features = {
        'spectral_centroid_mean': float(np.mean(spectral_centroids)),
        'spectral_centroid_std': float(np.std(spectral_centroids)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
        'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs],
        'mfcc_stds': [float(np.std(mfcc)) for mfcc in mfccs]
    }
    
    return features


def detect_sections_advanced(y, sr, beat_times):
    """
    Advanced section detection using multiple features.
    """
    # Use multiple features for segmentation
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    # Combine features
    features = np.vstack([chroma, mfcc, tonnetz])
    
    # Normalize features
    features = librosa.util.normalize(features, axis=1)
    
    # Use recurrence matrix for segmentation
    R = librosa.segment.recurrence_matrix(features, width=43, mode='affinity')
    
    # Determine cluster count based on song length (~1 boundary every ≈15 s, capped 4-20)
    duration_seconds = librosa.get_duration(y=y, sr=sr)
    est_k = max(4, min(20, int(duration_seconds // 15) or 4))

    try:
        boundaries = librosa.segment.agglomerative(features, k=est_k)
    except Exception:
        # Fallback: distance threshold segmentation as safety net
        boundaries = librosa.segment.agglomerative(features, distance_threshold=1.0)
    
    boundary_times = librosa.frames_to_time(boundaries, sr=sr)
    
    # Add start and end
    boundary_times = np.concatenate(([0.0], boundary_times, [librosa.get_duration(y=y, sr=sr)]))
    
    return boundary_times


def analyze_segments_detailed(y, sr, boundary_times, beat_times, tempo, key, mode):
    """
    Detailed analysis of each segment including loudness, tempo, key, and timbre.
    """
    segments = []
    
    for i in range(len(boundary_times) - 1):
        start_time = boundary_times[i]
        end_time = boundary_times[i + 1]
        duration = end_time - start_time
        
        # Extract segment audio
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_y = y[start_sample:end_sample]
        
        if len(segment_y) < 1024:  # Skip very short segments
            continue
        
        # Segment-specific analysis
        segment_loudness, _, _ = analyze_loudness(segment_y, sr)
        
        # Estimate segment tempo (simplified)
        try:
            segment_tempo, _ = librosa.beat.beat_track(y=segment_y, sr=sr)
            segment_tempo = float(segment_tempo)
            segment_tempo_confidence = 0.7  # Simplified confidence
        except Exception:
            segment_tempo = tempo
            segment_tempo_confidence = 0.3
        
        # Segment key analysis
        try:
            seg_key, _, seg_key_idx, seg_mode, seg_key_conf = estimate_key(segment_y, sr)
        except Exception:
            seg_key_idx = key
            seg_mode = mode
            seg_key_conf = 0.3
        
        # Time signature (inherit from track)
        time_signature = 4
        time_signature_confidence = 0.8
        
        segments.append({
            'start': float(start_time),
            'duration': float(duration),
            'confidence': 0.8,  # Default confidence
            'loudness': segment_loudness,
            'tempo': segment_tempo,
            'tempo_confidence': segment_tempo_confidence,
            'key': int(seg_key_idx),
            'key_confidence': seg_key_conf,
            'mode': int(seg_mode),
            'mode_confidence': 0.7,
            'time_signature': time_signature,
            'time_signature_confidence': time_signature_confidence
        })
    
    return segments


def analyze_bars_beats_tatums(y, sr, beat_times, tempo):
    """
    Analyze bars, beats, and tatums with confidence scores.
    """
    if len(beat_times) < 2:
        return [], [], []
    
    # Estimate beats per bar (assuming 4/4 time)
    beats_per_bar = 4
    
    # Create bars
    bars = []
    for i in range(0, len(beat_times) - beats_per_bar + 1, beats_per_bar):
        if i + beats_per_bar - 1 < len(beat_times):
            start = beat_times[i]
            end = beat_times[i + beats_per_bar - 1]
            duration = end - start
            
            bars.append({
                'start': float(start),
                'duration': float(duration),
                'confidence': 0.8
            })
    
    # Create beat objects
    beats = []
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        duration = beat_times[i + 1] - beat_times[i]
        
        beats.append({
            'start': float(start),
            'duration': float(duration),
            'confidence': 0.85
        })
    
    # Create tatums (subdivisions of beats)
    tatums = []
    tatum_subdivision = 2  # 2 tatums per beat
    
    for beat in beats:
        tatum_duration = beat['duration'] / tatum_subdivision
        for t in range(tatum_subdivision):
            tatum_start = beat['start'] + (t * tatum_duration)
            tatums.append({
                'start': float(tatum_start),
                'duration': float(tatum_duration),
                'confidence': 0.7
            })
    
    return bars, beats, tatums


def analyze_detailed_segments(y, sr, segment_boundaries):
    """
    Create detailed segments similar to Spotify's API with timbre and pitch analysis.
    """
    segments = []
    hop_length = 512
    
    # Pre-compute features for efficiency
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=hop_length)
    
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Create segments based on frame analysis
    segment_length = int(0.5 * sr / hop_length)  # ~0.5 second segments
    
    for i in range(0, len(rms), segment_length):
        start_frame = i
        end_frame = min(i + segment_length, len(rms))
        
        if end_frame - start_frame < segment_length // 2:  # Skip short segments
            continue
        
        start_time = frame_times[start_frame]
        duration = frame_times[end_frame - 1] - start_time if end_frame > start_frame else 0.1
        
        # Loudness analysis for this segment
        segment_rms = rms[start_frame:end_frame]
        loudness_start = librosa.amplitude_to_db(segment_rms[0]) if len(segment_rms) > 0 else -60
        loudness_max = librosa.amplitude_to_db(np.max(segment_rms)) if len(segment_rms) > 0 else -60
        loudness_end = librosa.amplitude_to_db(segment_rms[-1]) if len(segment_rms) > 0 else -60
        
        # Find max loudness time
        max_idx = np.argmax(segment_rms) if len(segment_rms) > 0 else 0
        loudness_max_time = (max_idx / len(segment_rms)) * duration if len(segment_rms) > 0 else 0
        
        # Pitch analysis (chroma)
        segment_chroma = chroma[:, start_frame:end_frame]
        pitches = np.mean(segment_chroma, axis=1) if segment_chroma.shape[1] > 0 else np.zeros(12)
        pitches = pitches / np.max(pitches) if np.max(pitches) > 0 else pitches
        
        # Timbre analysis (MFCC)
        segment_mfccs = mfccs[:, start_frame:end_frame]
        timbre = np.mean(segment_mfccs, axis=1) if segment_mfccs.shape[1] > 0 else np.zeros(12)
        
        segments.append({
            'start': float(start_time),
            'duration': float(duration),
            'confidence': 0.8,
            'loudness_start': float(loudness_start),
            'loudness_max': float(loudness_max),
            'loudness_max_time': float(loudness_max_time),
            'loudness_end': float(loudness_end),
            'pitches': pitches.tolist(),
            'timbre': timbre.tolist()
        })
    
    return segments


def label_segments_enhanced(segment_boundaries, y, sr, beat_times):
    """
    Enhanced segment labeling using multiple features and heuristics.
    """
    if len(segment_boundaries) <= 1:
        return [{"label": "main", "start": 0.0, "end": librosa.get_duration(y=y, sr=sr)}]

    labeled_segments = []
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Analyze each segment
    segment_features = []
    for i in range(len(segment_boundaries) - 1):
        start_time = segment_boundaries[i]
        end_time = segment_boundaries[i + 1]
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_y = y[start_sample:end_sample]
        
        if len(segment_y) > 1024:
            # Energy
            rms = np.mean(librosa.feature.rms(y=segment_y)[0])
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_y, sr=sr)[0])
            
            # Harmonic content
            harmonic, percussive = librosa.effects.hpss(segment_y)
            harmonic_ratio = np.mean(librosa.feature.rms(y=harmonic)[0]) / (rms + 1e-8)
            
            segment_features.append({
                'start': start_time,
                'end': end_time,
                'energy': rms,
                'spectral_centroid': spectral_centroid,
                'harmonic_ratio': harmonic_ratio
            })
    
    if not segment_features:
        return [{"label": "main", "start": 0.0, "end": total_duration}]
    
    # Calculate relative metrics
    energies = [s['energy'] for s in segment_features]
    max_energy = max(energies) if energies else 1
    
    # Label segments using enhanced heuristics
    for i, segment in enumerate(segment_features):
        label = "verse"  # Default
        
        # Intro/Outro detection
        if i == 0 and segment['energy'] < max_energy * 0.6:
            label = "intro"
        elif i == len(segment_features) - 1 and segment['energy'] < max_energy * 0.6:
            label = "outro"
        # High energy sections (chorus/drop)
        elif segment['energy'] >= max_energy * 0.85:
            label = "chorus"
        # Low energy sections (breakdown/bridge)
        elif segment['energy'] < max_energy * 0.4:
            label = "breakdown"
        # Mid energy with high harmonic content (verse)
        elif segment['harmonic_ratio'] > 0.6:
            label = "verse"
        # Mid energy with low harmonic content (bridge)
        else:
            label = "bridge"
        
        labeled_segments.append({
            "label": label,
            "start": segment['start'],
            "end": segment['end']
        })

    return labeled_segments


def analyze_song(file_path, include_lyrics=True):
    """
    Comprehensive audio analysis with Spotify-like features.
    """
    print(f"🎵 Analyzing: {file_path}...")
    
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        print(f"  ✓ Loaded audio: {duration:.2f}s at {sr}Hz")
        
        # Basic track info
        track_info = {
            'num_samples': len(y),
            'duration': float(duration),
            'sample_md5': "",  # Not implemented
            'offset_seconds': 0,
            'window_seconds': 0,
            'analysis_sample_rate': int(sr),
            'analysis_channels': 1,
            'end_of_fade_in': 0.0,  # Could be enhanced
            'start_of_fade_out': float(duration * 0.95)  # Estimate
        }
        
        # Tempo and beat analysis
        print("  - Analyzing tempo and beats...")
        tempo, beat_times, tempo_confidence = analyze_tempo_and_beats(y, sr)

        # Key analysis
        print("  - Analyzing key...")
        key_standard, key_camelot, key_idx, mode, key_confidence = estimate_key(y, sr)
        
        # Time signature
        print("  - Analyzing time signature...")
        time_signature, time_signature_confidence = analyze_time_signature(y, sr, tempo)
        
        # Loudness analysis
        print("  - Analyzing loudness...")
        loudness, peak_loudness, loudness_range = analyze_loudness(y, sr)
    
        # Spectral features for energy estimation
        print("  - Extracting spectral features...")
        spectral_features = analyze_spectral_features(y, sr)
        
        # Normalize energy (using spectral centroid and RMS)
        rms_energy = np.mean(librosa.feature.rms(y=y)[0])
        normalized_energy = min(1.0, rms_energy * 2)  # Scale to 0-1
        
        # Section detection
        print("  - Detecting sections...")
        section_boundaries = detect_sections_advanced(y, sr, beat_times)
        
        # Detailed segments (similar to Spotify segments)
        print("  - Analyzing detailed segments...")
        detailed_segments = analyze_detailed_segments(y, sr, section_boundaries)
        
        # Bars, beats, tatums
        print("  - Analyzing rhythmic structure...")
        bars, beats, tatums = analyze_bars_beats_tatums(y, sr, beat_times, tempo)
        
        # Section analysis with metadata
        sections = analyze_segments_detailed(y, sr, section_boundaries, beat_times, tempo, key_idx, mode)
        
        # Enhanced structural labeling
        structural_analysis = label_segments_enhanced(section_boundaries, y, sr, beat_times)

        # Lyrics transcription
        lyrics_timed = []
        if include_lyrics:
            print("  - Transcribing lyrics...")
            try:
                client = OpenAI()
                with open(file_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )
                
                if hasattr(transcription, 'words') and transcription.words:
                    lyrics_timed = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end
                        }
                        for word in transcription.words
                    ]
                    print(f"    ✓ Transcribed {len(lyrics_timed)} words")
                else:
                    print("    - No word-level timestamps available")
            except Exception as e:
                print(f"    - Transcription failed: {e}")
        
        # Compile comprehensive analysis
        analysis_data = {
            # Meta information
            "meta": {
                "analyzer_version": "2.0.0",
                "platform": "Python",
                "detailed_status": "OK",
                "status_code": 0,
                "timestamp": int(librosa.get_duration(y=y, sr=sr)),
                "analysis_time": 0.0,  # Could be measured
                "input_process": f"librosa {sr}Hz"
            },
            
            # Track-level analysis
            "track": {
                **track_info,
                "loudness": loudness,
                "tempo": float(tempo),
                "tempo_confidence": tempo_confidence,
                "time_signature": time_signature,
                "time_signature_confidence": time_signature_confidence,
                "key": key_idx,
                "key_confidence": key_confidence,
                "mode": mode,
                "mode_confidence": 0.8,  # Could be enhanced
                # Placeholder strings for compatibility
                "codestring": "",
                "code_version": 1.0,
                "echoprintstring": "",
                "echoprint_version": 1.0,
                "synchstring": "",
                "synch_version": 1.0,
                "rhythmstring": "",
                "rhythm_version": 1.0
            },
            
            # Rhythmic analysis
            "bars": bars,
            "beats": beats,
            "tatums": tatums,
            
            # Structural analysis
            "sections": sections,
            "segments": detailed_segments,
            
            # Legacy format for compatibility
            "file_path": file_path,
            "bpm": str(round(tempo, 2)),
            "key_standard": key_standard,
            "key_camelot": key_camelot,
            "energy_normalized": str(round(normalized_energy, 3)),
            "beat_grid_seconds": beat_times.tolist(),
            "structural_analysis": structural_analysis,
            "lyrics_timed": lyrics_timed,
            
            # Additional features
            "spectral_features": spectral_features,
            "loudness_range": loudness_range,
            "peak_loudness": peak_loudness
        }
    
        print("  ✓ Analysis complete!")
        return analysis_data
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced audio analysis with Spotify-like features.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file")
    parser.add_argument("-o", "--output_file", type=str, help="Output JSON file path")
    parser.add_argument("--no-lyrics", action="store_true", help="Skip lyrics transcription")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found at {args.input_file}")
        exit(1)
    
    # Analyze the song
    include_lyrics = not args.no_lyrics
    analysis_result = analyze_song(args.input_file, include_lyrics=include_lyrics)
    
    if analysis_result:
        # Determine output path
        output_path = args.output_file
        if not output_path:
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            output_path = f"analysis/{base_name}_analysis.json"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
            
        print(f"\n✅ Analysis saved to: {output_path}")
        
        # Print summary
        track = analysis_result['track']
        print(f"\n📊 Analysis Summary:")
        print(f"   🎵 Duration: {track['duration']:.1f}s")
        print(f"   🥁 BPM: {track['tempo']:.1f} (confidence: {track['tempo_confidence']:.2f})")
        print(f"   🎹 Key: {analysis_result['key_standard']} ({analysis_result['key_camelot']})")
        print(f"   📏 Time Signature: {track['time_signature']}/4")
        print(f"   🔊 Loudness: {track['loudness']:.1f} dB")
        print(f"   ⚡ Energy: {float(analysis_result['energy_normalized']):.2f}")
        print(f"   🏗️  Sections: {len(analysis_result['structural_analysis'])}")
        print(f"   🎤 Lyrics: {len(analysis_result['lyrics_timed'])} words")
    else:
        print("❌ Analysis failed!")
        exit(1)
