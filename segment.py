#!/usr/bin/env python3
"""segment_track.py

Utility script for quick testing of the HPSS segmentation used in controller.py.
Given one input audio file (e.g. family_ties.mp3) it outputs two WAV files:
    <basename>_beat.wav      – percussive / drums stem
    <basename>_vocals.wav    – harmonic / vocal/melodic stem

Example:
    python segment_track.py songs/family_ties.mp3 -o stems/
"""
import argparse
import os
import librosa
import soundfile as sf


def separate_stems(path: str, out_dir: str | None = None, sr: int = 44100) -> tuple[str, str]:
    """Load *path*, perform HPSS, save stems in *out_dir*.

    Returns (beat_path, vocal_path).
    """
    if out_dir is None:
        out_dir = os.path.dirname(path)

    basename = os.path.splitext(os.path.basename(path))[0]
    beat_path = os.path.join(out_dir, f"{basename}_beat.wav")
    vocal_path = os.path.join(out_dir, f"{basename}_vocals.wav")

    print(f"Loading {path} …")
    y, _ = librosa.load(path, sr=sr, mono=True)
    print("Running HPSS (this may take a few seconds) …")
    harmonic, percussive = librosa.effects.hpss(y)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing percussive stem → {beat_path}")
    sf.write(beat_path, percussive, sr)
    print(f"Writing harmonic stem   → {vocal_path}")
    sf.write(vocal_path, harmonic, sr)
    print("Done.")
    return beat_path, vocal_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split one track into beat and vocals using HPSS.")
    parser.add_argument("input_file", help="Path to an audio file (mp3, wav, etc.)")
    parser.add_argument("-o", "--output_dir", help="Directory to save the stems (default: alongside input)")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample-rate for loading & saving (default 44100)")
    args = parser.parse_args()

    separate_stems(args.input_file, args.output_dir, sr=args.sr)


if __name__ == "__main__":
    main() 
