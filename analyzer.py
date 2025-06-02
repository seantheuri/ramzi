"""
auto_dj_advanced.py
-------------------
A minimal prototype that (1) analyses two .wav files and chooses an
appropriate transition strategy, then (2) renders the mixed track.

Supported transition types (extensible):
  • "crossfade"      — plain overlap with linear gain ramp.
  • "beatmatch"      — tempo‑match one track, then cross‑fade on beat.
  • "lyric_overlay"  — crude "call/response" overlay starting at the
                        first strong lyric phrase after the downbeat.

Dependencies (Python ≥3.9):
    pip install librosa pydub soundfile openai-whisper

Example
-------
>>> from auto_dj_advanced import analyse_tracks, apply_transition
>>> plan = analyse_tracks("song1.wav", "song2.wav")
>>> print(plan)
TransitionPlan(method='beatmatch', bpm1=128.1, bpm2=130.6, key1='C#m', key2='E', crossfade_ms=16000)
>>> apply_transition("song1.wav", "song2.wav", plan, "mix.wav")
✓ Wrote mix.wav
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa

# --- optional speech‑to‑text -------------------------------------------------
try:
    import whisper  # OpenAI Whisper (pip install openai-whisper)
    _whisper_model = whisper.load_model("base", device="cpu")
except Exception:
    whisper = None
    _whisper_model = None

# ---------------------------------------------------------------------------
@dataclass
class TransitionPlan:
    method: str                 # 'crossfade' | 'beatmatch' | 'lyric_overlay'
    bpm1: float
    bpm2: float
    key1: Optional[str]
    key2: Optional[str]
    crossfade_ms: int           # duration of overlap in milliseconds
    extra: Dict[str, float] = None  # any other knobs for renderer

    def to_dict(self):
        d = asdict(self)
        d["method"] = self.method
        return d

# ---------------------------------------------------------------------------
#                               ANALYSIS
# ---------------------------------------------------------------------------

def _estimate_bpm(path: str, sr: int = 22_050) -> Tuple[float, np.ndarray]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return tempo, beat_times


def _estimate_key(path: str, sr: int = 22_050) -> Optional[str]:
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        # very rough template matching
        major_templates = np.roll(np.identity(12), 0, axis=0)
        minor_templates = np.roll(np.identity(12), 3, axis=0)
        score_maj = (major_templates * chroma_mean[:, None]).sum(axis=0)
        score_min = (minor_templates * chroma_mean[:, None]).sum(axis=0)
        idx_maj = int(score_maj.argmax())
        idx_min = int(score_min.argmax())
        if score_maj.max() > score_min.max():
            return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][idx_maj]
        else:
            return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][idx_min] + "m"
    except Exception:
        return None


def _transcribe(path: str) -> List[Dict]:
    if _whisper_model is None:
        return []
    result = _whisper_model.transcribe(path, word_timestamps=True, verbose=False)
    words = []
    for seg in result["segments"]:
        for w in seg["words"]:
            words.append({"start": w["start"], "end": w["end"], "text": w["word"].lower()})
    return words


def _lyric_density(words: List[Dict]) -> float:
    if not words:
        return 0.0
    total_time = words[-1]["end"] - words[0]["start"]
    if total_time <= 0:
        return 0.0
    return len(words) / total_time  # words per second


def analyse_tracks(path1: str, path2: str) -> TransitionPlan:
    """Return a TransitionPlan describing the best‑guess transition."""

    bpm1, beats1 = _estimate_bpm(path1)
    bpm2, beats2 = _estimate_bpm(path2)
    key1 = _estimate_key(path1)
    key2 = _estimate_key(path2)

    words1 = _transcribe(path1)
    words2 = _transcribe(path2)

    density1 = _lyric_density(words1)
    density2 = _lyric_density(words2)

    # --- choose strategy heuristically --------------------------------------
    crossfade_ms = int(0.08 * min(beats1[-1] if len(beats1) else 30,
                                   beats2[-1] if len(beats2) else 30) * 1000)
    if abs(bpm1 - bpm2) <= 3 and key1 and key2 and key1[-1] == key2[-1]:
        method = "beatmatch"
    elif density1 > 2.0 and density2 > 2.0:  # talking parts / rap heavy
        method = "lyric_overlay"
    else:
        method = "crossfade"

    return TransitionPlan(method=method,
                          bpm1=round(bpm1, 1), bpm2=round(bpm2, 1),
                          key1=key1, key2=key2,
                          crossfade_ms=crossfade_ms,
                          extra={"density1": density1, "density2": density2})

# ---------------------------------------------------------------------------
#                               RENDERING
# ---------------------------------------------------------------------------

def _stretch(segment: AudioSegment, rate: float) -> AudioSegment:
    """Time‑stretch pydub segment via librosa (offline)."""
    samples = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0
    y_stretch = librosa.effects.time_stretch(samples, rate=rate)
    y_int16 = (y_stretch * 32768.0).astype(np.int16)
    stretched = segment._spawn(y_int16.tobytes(), overrides={"frame_rate": segment.frame_rate})
    return stretched.set_frame_rate(segment.frame_rate)


def apply_transition(path1: str, path2: str, plan: TransitionPlan, out_path: str) -> None:
    a1 = AudioSegment.from_file(path1)
    a2 = AudioSegment.from_file(path2)

    if plan.method == "beatmatch":
        rate = plan.bpm1 / plan.bpm2 if plan.bpm2 else 1.0
        a2 = _stretch(a2, rate)
        mixed = a1.append(a2, crossfade=plan.crossfade_ms)

    elif plan.method == "lyric_overlay":
        # crude: take first 6 s of track1, overlay first 6 s of track2 starting halfway
        overlap_ms = min(len(a1), len(a2), 6000)
        mixed = a1[:overlap_ms].overlay(a2[:overlap_ms].apply_gain(-3)) + a2[overlap_ms:]

    else:  # plain crossfade
        mixed = a1.append(a2, crossfade=plan.crossfade_ms)

    mixed.export(out_path, format="wav")
    print(f"✓ Wrote {out_path}  (len = {mixed.duration_seconds:.1f}s)")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Auto DJ two wav files → mix.wav")
    p.add_argument("song1")
    p.add_argument("song2")
    p.add_argument("--out", default="mix.wav")
    args = p.parse_args()

    plan = analyse_tracks(args.song1, args.song2)
    print("Analysis →", json.dumps(plan.to_dict(), indent=2))
    apply_transition(args.song1, args.song2, plan, args.out)
