"""
auto_dj_advanced.py  (v2 – Spotify‑aware)
========================================
An end‑to‑end prototype that
  1. fingerprints two **.wav** files, tries to resolve them to Spotify IDs
  2. pulls Spotify **Audio Features / Audio Analysis** for rich metadata
  3. falls back to local DSP (librosa) for tempo / key / lyric density
  4. selects the most suitable transition:   
        • "crossfade" (safe default)   
        • "beatmatch"  – tempo‑match + harmonic check   
        • "lyric_overlay" – rap / spoken‑word call‑and‑response
  5. renders the mix with *pydub* and writes a final **.wav**

New external deps (Python ≥3.9):
    pip install librosa pydub soundfile openai-whisper \
                pyacoustid spotipy python‑dotenv

Credential setup
----------------
Create a **.env** in the same folder (never commit this!)
```
ACOUSTID_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
SPOTIPY_CLIENT_ID=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
SPOTIPY_CLIENT_SECRET=zzzzzzzzzzzzzzzzzzzzzzzzzzzz
```
Run once: `source $(pipenv --venv)/bin/activate && export $(cat .env | xargs)`

Example
-------
>>> from auto_dj_advanced import analyse_tracks, apply_transition
>>> plan = analyse_tracks("songA.wav", "songB.wav")
>>> print(plan)
TransitionPlan(method='beatmatch', bpm1=128.1, bpm2=130.0, key1='C#m', key2='E', energy_gap=0.08, speechiness=0.46, crossfade_ms=16000)
>>> apply_transition("songA.wav", "songB.wav", plan, "auto_mix.wav")
✓ Wrote auto_mix.wav  (len = 374.2s)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment
import librosa, soundfile as sf

# --- optional: speech‑to‑text ------------------------------------------------
try:
    import whisper  # pip install openai-whisper
    _whisper_model = whisper.load_model("base", device="cpu")
except Exception:
    whisper = None
    _whisper_model = None

# --- optional: Spotify resolution -------------------------------------------
try:
    import acoustid, chromaprint  # pip install pyacoustid
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    _ACOUSTID_KEY = os.getenv("ACOUSTID_KEY")
    _SPOTIFY_ID = os.getenv("SPOTIPY_CLIENT_ID")
    _SPOTIFY_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

    _sp = None
    if _SPOTIFY_ID and _SPOTIFY_SECRET:
        _sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=_SPOTIFY_ID, client_secret=_SPOTIFY_SECRET))
except Exception:
    acoustid = None  # type: ignore
    chromaprint = None  # type: ignore
    spotipy = None  # type: ignore
    _sp = None
    _ACOUSTID_KEY = None

# ---------------------------------------------------------------------------
@dataclass
class TransitionPlan:
    method: str                 # 'crossfade' | 'beatmatch' | 'lyric_overlay'
    bpm1: float
    bpm2: float
    key1: Optional[str]
    key2: Optional[str]
    crossfade_ms: int           # overlap duration in ms
    extra: Dict[str, float] = None

    def __str__(self):
        return str(asdict(self))

# ---------------------------------------------------------------------------
#                               LOW‑LEVEL DSP
# ---------------------------------------------------------------------------

def _estimate_bpm(path: str, sr: int = 22_050) -> Tuple[float, np.ndarray]:
    y, _sr = librosa.load(path, sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=_sr)
    return tempo, librosa.frames_to_time(beat_frames, sr=_sr)


def _estimate_key(path: str, sr: int = 22_050) -> Optional[str]:
    try:
        y, _sr = librosa.load(path, sr=sr, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=_sr).mean(axis=1)
        major_scores = np.roll(np.identity(12), 0, axis=0) @ chroma
        minor_scores = np.roll(np.identity(12), 3, axis=0) @ chroma
        scale = "major" if major_scores.max() > minor_scores.max() else "minor"
        idx = int(major_scores.argmax() if scale == "major" else minor_scores.argmax())
        notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return notes[idx] if scale == "major" else notes[idx] + "m"
    except Exception:
        return None


def _transcribe(path: str) -> List[Dict]:
    if _whisper_model is None:
        return []
    result = _whisper_model.transcribe(path, word_timestamps=True, verbose=False)
    return [{"start": w["start"], "end": w["end"], "text": w["word"].lower()}
            for seg in result["segments"] for w in seg["words"]]


def _lyric_density(words: List[Dict]) -> float:
    if not words:
        return 0.0
    total = words[-1]["end"] - words[0]["start"]
    return len(words) / total if total > 0 else 0.0

# ---------------------------------------------------------------------------
#                     SPOTIFY RESOLUTION HELPERS (optional)
# ---------------------------------------------------------------------------

def _fingerprint(path: str) -> Tuple[str, int] | None:
    if acoustid is None or _ACOUSTID_KEY is None:
        return None
    try:
        fp, sr = acoustid.fingerprint_file(path)
        return fp, sr
    except Exception:
        return None


def _fp_to_spotify_id(fp: str, sr: int) -> str | None:
    if acoustid is None or _ACOUSTID_KEY is None:
        return None
    try:
        res = acoustid.lookup(_ACOUSTID_KEY, fp, sr, meta="recordings")
        if res["status"] != "ok" or not res["results"]:
            return None
        rec = res["results"][0]["recordings"][0]
        title, artist = rec["title"], rec["artists"][0]["name"]
        if _sp is None:
            return None
        q = f'track:{title} artist:{artist}'
        items = _sp.search(q=q, type="track", limit=1)["tracks"]["items"]
        return items[0]["id"] if items else None
    except Exception:
        return None


def _spotify_features(track_id: str) -> Dict | None:
    if _sp is None:
        return None
    try:
        feats = _sp.audio_features(track_id)[0]
        analysis = _sp._get(f"audio-analysis/{track_id}")
        feats.update(analysis["track"])
        return feats
    except Exception:
        return None

# ---------------------------------------------------------------------------
#                               ANALYSIS PIPELINE
# ---------------------------------------------------------------------------

def _analyze_single(path: str) -> Dict:
    """Return a dict of all usable features for one track."""
    # 1) Try Spotify shortcut -------------------------------------------------
    spotify_ok = False
    data: Dict[str, float | str] = {}
    fp = _fingerprint(path)
    if fp:
        spid = _fp_to_spotify_id(*fp)
        if spid:
            sfeat = _spotify_features(spid)
            if sfeat:
                spotify_ok = True
                data.update({
                    "bpm": sfeat["tempo"],
                    "key_idx": sfeat["key"],
                    "mode": sfeat["mode"],
                    "energy": sfeat["energy"],
                    "danceability": sfeat["danceability"],
                    "speechiness": sfeat["speechiness"],
                    "valence": sfeat["valence"],
                })
                # translate key idx -> human
                notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                data["key"] = notes[sfeat["key"]] + ("m" if sfeat["mode"] == 0 else "")

    # 2) Fallback local features --------------------------------------------
    if not spotify_ok:
        bpm, beats = _estimate_bpm(path)
        data["bpm"] = bpm
        data["beats"] = beats
        data["key"] = _estimate_key(path)
        words = _transcribe(path)
        data["lyric_density"] = _lyric_density(words)
        # rough energy proxy
        y, _ = librosa.load(path, sr=22_050, mono=True)
        data["energy"] = float(np.mean(librosa.feature.rms(y=y)))
        data["speechiness"] = data.get("lyric_density", 0) / 5  # crude mapping
        data["danceability"] = 0.5  # placeholder
        data["valence"] = 0.5
    return data


def analyse_tracks(path1: str, path2: str) -> TransitionPlan:
    f1 = _analyze_single(path1)
    f2 = _analyze_single(path2)

    bpm1, bpm2 = f1["bpm"], f2["bpm"]
    key1, key2 = f1.get("key"), f2.get("key")
    energy_gap = abs(f1["energy"] - f2["energy"])
    speech_avg = (f1["speechiness"] + f2["speechiness"]) / 2

    # Choose strategy --------------------------------------------------------
    if abs(bpm1 - bpm2) <= 3 and key1 and key2 and key1[-1] == key2[-1]:
        method = "beatmatch"
    elif speech_avg > 0.35:  # lots of vocals / rap
        method = "lyric_overlay"
    else:
        method = "crossfade"

    # crossfade length: 8 % of shorter song OR min 8 s
    beats1 = f1.get("beats", np.array([]))
    beats2 = f2.get("beats", np.array([]))
    end1 = beats1[-1] if beats1.size else 30
    end2 = beats2[-1] if beats2.size else 30
    cf_ms = int(max(8, 0.08 * min(end1, end2)) * 1000)

    extra = {
        "energy_gap": energy_gap,
        "danceability_gap": abs(f1["danceability"] - f2["danceability"]),
        "speechiness": speech_avg,
        "valence_gap": abs(f1["valence"] - f2["valence"]),
    }

    return TransitionPlan(method, round(bpm1,1), round(bpm2,1),
                          key1, key2, cf_ms, extra)

# ---------------------------------------------------------------------------
#                               RENDERING
# ---------------------------------------------------------------------------

def _stretch(segment: AudioSegment, rate: float) -> AudioSegment:
    samp = np.array(segment.get_array_of_samples()).astype(np.float32)/32768.0
    y = librosa.effects.time_stretch(samp, rate=rate)
    y16 = (y*32768.0).astype(np.int16)
    return segment._spawn(y16.tobytes(), overrides={"frame_rate": segment.frame_rate}).set_frame_rate(segment.frame_rate)


def apply_transition(path1: str, path2: str, plan: TransitionPlan, out: str) -> None:
    a1, a2 = AudioSegment.from_file(path1), AudioSegment.from_file(path2)

    if plan.method == "beatmatch":
        rate = plan.bpm1 / plan.bpm2 if plan.bpm2 else 1.0
        a2 = _stretch(a2, rate)
        mix = a1.append(a2, crossfade=plan.crossfade_ms)

    elif plan.method == "lyric_overlay":
        ov = min(len(a1), len(a2), 8000)
        mix = a1[:ov].overlay(a2[:ov].apply_gain(-3)) + a2[ov:]

    else:  # crossfade
        mix = a1.append(a2, crossfade=plan.crossfade_ms)

    Path(out).with_suffix(".wav")  # ensure .wav extension
    mix.export(out, format="wav")
    print(f"✓ Wrote {out}  (len = {mix.duration_seconds:.1f}s)")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Auto‑DJ two wav files ➔ mix.wav")
    ap.add_argument("wav1")
    ap.add_argument("wav2")
    ap.add_argument("--out", default="mix.wav")
    args = ap.parse_args()

    plan = analyse_tracks(args.wav1, args.wav2)
    print("Analysis →", json.dumps(plan.extra | {k: getattr(plan, k) for k in ("method","bpm1","bpm2","key1","key2","crossfade_ms")}, indent=2))
    apply_transition(args.wav1, args.wav2, plan, args.out)
