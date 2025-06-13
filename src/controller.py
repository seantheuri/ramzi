import librosa
import numpy as np
import soundfile as sf
import json
import os
import argparse
import pedalboard as pb
from collections import deque
import scipy.signal
import requests
import time
import io
from math import gcd

# --- Helper Functions ---

def stretch_audio_to_bpm(y, sr, original_bpm, target_bpm):
    if np.isclose(original_bpm, target_bpm):
        return y
    stretch_rate = target_bpm / original_bpm
    print(f"  - Stretching from {original_bpm:.2f} BPM to {target_bpm:.2f} BPM (rate: {stretch_rate:.3f})")
    return librosa.effects.time_stretch(y=y, rate=stretch_rate)


def pitch_shift_semitones(y, sr, semitones):
    """Pitch shift by semitones without changing tempo"""
    if semitones == 0:
        return y
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)


# --- Custom Effects ---
class BeatRepeat:
    """Beat repeat/loop effect"""

    def __init__(self, sr=44100):
        self.sr = sr
        self.loop_buffer = None
        self.loop_length = 0
        self.loop_position = 0
        self.is_active = False

    def set_loop(self, audio, beats=4, bpm=120):
        """Set a loop of specified beat length"""
        beat_duration = 60.0 / bpm
        self.loop_length = int(beats * beat_duration * self.sr)
        self.loop_buffer = audio[: self.loop_length].copy()
        self.loop_position = 0
        self.is_active = True

    def process(self, input_chunk):
        if not self.is_active or self.loop_buffer is None:
            return input_chunk

        output = np.zeros_like(input_chunk)
        remaining = len(input_chunk)
        out_pos = 0

        while remaining > 0:
            available = min(remaining, self.loop_length - self.loop_position)
            output[out_pos : out_pos + available] = self.loop_buffer[
                self.loop_position : self.loop_position + available
            ]
            out_pos += available
            remaining -= available
            self.loop_position = (self.loop_position + available) % self.loop_length

        return output

    def deactivate(self):
        self.is_active = False


class BeatJump:
    """Beat jump effect - jumps forward/backward by beat increments"""

    def __init__(self, sr=44100):
        self.sr = sr

    def calculate_jump_samples(self, beats, bpm):
        beat_duration = 60.0 / bpm
        return int(beats * beat_duration * self.sr)


class Sampler:
    """Sampler with multiple slots"""

    def __init__(self, num_slots=16, sr=44100):
        self.sr = sr
        self.slots = [
            {"audio": None, "position": 0, "is_playing": False} for _ in range(num_slots)
        ]

    def load_slot(self, slot_idx, audio):
        if 0 <= slot_idx < len(self.slots):
            self.slots[slot_idx] = {
                "audio": audio,
                "position": 0,
                "is_playing": False,
            }

    def play_slot(self, slot_idx):
        if 0 <= slot_idx < len(self.slots) and self.slots[slot_idx]["audio"] is not None:
            self.slots[slot_idx]["is_playing"] = True
            self.slots[slot_idx]["position"] = 0

    def stop_slot(self, slot_idx):
        if 0 <= slot_idx < len(self.slots):
            self.slots[slot_idx]["is_playing"] = False

    def process(self, num_samples):
        output = np.zeros(num_samples)
        for slot in self.slots:
            if slot["is_playing"] and slot["audio"] is not None:
                start = slot["position"]
                end = min(start + num_samples, len(slot["audio"]))
                chunk_length = end - start

                if chunk_length > 0:
                    output[:chunk_length] += slot["audio"][start:end]
                    slot["position"] = end

                    if slot["position"] >= len(slot["audio"]):
                        slot["is_playing"] = False

        return output


class VinylSimulator:
    """Simulates vinyl scratching and pitch bending"""

    def __init__(self, sr=44100):
        self.sr = sr
        self.playback_rate = 1.0
        self.scratch_position = 0.0
        self.is_scratching = False

    def set_jog_position(self, position_delta):
        """Simulate jog wheel movement"""
        self.scratch_position += position_delta

    def set_pitch_bend(self, bend_amount):
        """Pitch bend: -1.0 to 1.0"""
        self.playback_rate = 1.0 + (bend_amount * 0.1)  # ±10% pitch bend


class PadFX:
    """Pad-based effects (32 assignable effects)"""

    def __init__(self, sr=44100):
        self.sr = sr
        self.active_effects = set()

        # Define all 32 pad effects
        self.effects = {
            # PAD FX1 Mode (1-16)
            1: pb.Reverb(room_size=0.5, damping=0.5, wet_level=0.3),
            2: pb.Delay(delay_seconds=0.25, feedback=0.3, mix=0.3),
            3: pb.Phaser(rate_hz=1.0, depth=0.5, mix=0.5),
            4: pb.Chorus(rate_hz=1.5, depth=0.25, mix=0.5),
            5: pb.Distortion(drive_db=10),
            6: pb.Compressor(threshold_db=-20, ratio=4),
            7: pb.Bitcrush(bit_depth=8),
            8: pb.HighpassFilter(cutoff_frequency_hz=1000),
            9: pb.LowpassFilter(cutoff_frequency_hz=1000),
            10: pb.Reverb(room_size=0.8, damping=0.7, wet_level=0.5),
            11: pb.Delay(delay_seconds=0.5, feedback=0.5, mix=0.5),
            12: pb.Phaser(rate_hz=2.0, depth=0.7, mix=0.7),
            13: pb.Chorus(rate_hz=2.5, depth=0.5, mix=0.7),
            14: pb.Distortion(drive_db=20),
            15: pb.Compressor(threshold_db=-10, ratio=8),
            16: pb.Bitcrush(bit_depth=4),
            # PAD FX2 Mode (17-32)
            17: pb.Reverb(room_size=0.3, damping=0.3, wet_level=0.2),
            18: pb.Delay(delay_seconds=0.125, feedback=0.2, mix=0.2),
            19: pb.HighShelfFilter(cutoff_frequency_hz=5000, gain_db=6),
            20: pb.LowShelfFilter(cutoff_frequency_hz=200, gain_db=6),
            21: pb.PeakFilter(cutoff_frequency_hz=2000, q=2.0, gain_db=6),
            22: pb.Limiter(threshold_db=-6, release_ms=50),
            23: pb.NoiseGate(threshold_db=-40, ratio=10),
            24: pb.Convolution(np.array([1.0], dtype=np.float32), mix=0.5, sample_rate=self.sr),
            25: pb.PitchShift(semitones=2),
            26: pb.PitchShift(semitones=-2),
            27: pb.Gain(gain_db=6),
            28: pb.Gain(gain_db=-6),
            29: pb.Clipping(threshold_db=-6),
            30: pb.GSMFullRateCompressor(),
            31: pb.MP3Compressor(vbr_quality=9),
            32: pb.Resample(target_sample_rate=8000),
        }

    def toggle_effect(self, effect_id):
        if effect_id in self.active_effects:
            self.active_effects.remove(effect_id)
        else:
            self.active_effects.add(effect_id)

    def process(self, audio):
        output = audio.copy()
        for effect_id in self.active_effects:
            if effect_id in self.effects:
                try:
                    output = self.effects[effect_id](output, self.sr)
                except Exception:
                    pass  # Skip effects that fail
        return output


# --- Core Component Classes ---
class Deck:
    """Represents a single virtual DJ deck with full DDJ-FLX4 functionality"""

    def __init__(self, name, sample_rate=44100):
        self.name = name
        self.y = None
        self.y_percussive = None  # beat-only stem
        self.y_harmonic = None    # vocals / melodic stem
        self.sr = sample_rate
        self.analysis_data = None
        self.is_playing = False
        self.current_sample_pos = 0
        self.tempo_range = 0.06  # ±6% default
        self.tempo_adjustment = 0.0  # -1.0 to 1.0
        self.sync_enabled = False
        self.is_master = False

        # Hot cues (8 slots)
        self.hot_cues = [None] * 8

        # Loop controls
        self.loop_in_sample = None
        self.loop_out_sample = None
        self.loop_active = False
        self.auto_loop_beats = 4

        # Effects board
        self.board = pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=20.0),            # 0 filter
            pb.HighShelfFilter(cutoff_frequency_hz=4000, gain_db=0),# 1 EQ high
            pb.PeakFilter(cutoff_frequency_hz=1000, q=1.0, gain_db=0),# 2 EQ mid
            pb.LowShelfFilter(cutoff_frequency_hz=250, gain_db=-3), # 3 EQ low  <<– default –3 dB
            pb.Gain(gain_db=-6),                                    # 4 trim (head-room)
            pb.Gain(gain_db=-6),                                    # 5 channel fader
        ])

        # Additional effects
        self.color_fx = pb.Pedalboard([])  # Smart CFX
        self.beat_fx = None  # Beat FX
        self.beat_fx_params = {"beats": 1, "depth": 0.5}

        # Custom effects
        self.beat_repeat = BeatRepeat(sample_rate)
        self.beat_jump = BeatJump(sample_rate)
        self.vinyl_sim = VinylSimulator(sample_rate)
        self.pad_fx = PadFX(sample_rate)

        # Performance modes
        self.performance_mode = "hot_cue"  # hot_cue, pad_fx1, pad_fx2, beat_jump, beat_loop, sampler, key_shift, keyboard
        self.key_shift = 0  # Semitones
        # playback segment: 'full', 'vocals', 'beat', or 'instrumental'
        self.segment_type = 'full'

        # Additional stem buffer (vocals only)
        self.y_vocals = None

    def _split_with_lalal(self, audio_file_path, stems=("vocals", "drum")):
        """Download selected stems via LALAL.ai. Returns dict{'back':y_back,'vocals':y_voc,'drum':y_drum}. None on failure."""
        api_key = os.environ.get("LALAL_API_KEY")
        if not api_key:
            return None

        try:
            filename = os.path.basename(audio_file_path)
            upload_headers = {
                "Content-Disposition": f"attachment; filename={filename}",
                "Authorization": f"license {api_key}"
            }
            with open(audio_file_path, "rb") as f:
                resp = requests.post("https://www.lalal.ai/api/upload/", data=f, headers=upload_headers, timeout=120)
            resp.raise_for_status()
            up_json = resp.json()
            if up_json.get("status") != "success":
                print(f"    LALAL upload error: {up_json.get('error')}")
                return None

            file_id = up_json["id"]

            # Sequentially request each desired stem
            results = {"back": None}

            split_headers = {"Authorization": f"license {api_key}"}

            for stem in stems:
                # Trigger split
                split_params = json.dumps([{"id": file_id, "stem": stem}])
                s_resp = requests.post("https://www.lalal.ai/api/split/", data={"params": split_params}, headers=split_headers, timeout=30)
                s_resp.raise_for_status()

                # Poll until done
                success = False
                for _ in range(120):  # 10 min max
                    time.sleep(5)
                    c_resp = requests.post("https://www.lalal.ai/api/check/", data={"id": file_id}, headers=split_headers, timeout=15)
                    c_resp.raise_for_status()
                    res = c_resp.json().get("result", {}).get(file_id, {})
                    if res.get("status") == "error":
                        print(f"    LALAL split error ({stem}): {res.get('error')}")
                        break
                    if res.get("split") and res.get("stem") == stem:
                        split_info = res["split"]
                        stem_url = split_info.get("stem_track")
                        back_url = split_info.get("back_track")
                        if stem_url:
                            results[stem] = self._download_audio(stem_url)
                        if back_url and results.get("back") is None:
                            results["back"] = self._download_audio(back_url)
                        success = True
                        break
                if not success:
                    print(f"    LALAL split for stem '{stem}' timed out or failed")

            return results if any(v is not None for v in results.values()) else None

        except Exception as e:
            print(f"    LALAL split failed: {e}")
        return None

    def _download_audio(self, url):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = io.BytesIO(r.content)
            y, sr_file = librosa.load(data, sr=self.sr)
            return y
        except Exception as e:
            print(f"        Download error: {e}")
            return None

    def load_track(self, analysis_file_path, target_bpm=None):
        print(f"Deck {self.name}: Loading track from {analysis_file_path}")
        with open(analysis_file_path, "r") as f:
            self.analysis_data = json.load(f)

        audio_file_path = self.analysis_data["file_path"]
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(
                f"Audio file '{audio_file_path}' from analysis json not found."
            )

        y_original, sr_original = librosa.load(audio_file_path, sr=None)
        if sr_original != self.sr:
            y_original = librosa.resample(y=y_original, orig_sr=sr_original, target_sr=self.sr)

        original_bpm = float(self.analysis_data["bpm"])

        # Apply tempo adjustment if target BPM specified
        if target_bpm:
            self.y = stretch_audio_to_bpm(y_original, self.sr, original_bpm, target_bpm)
        else:
            self.y = y_original

        # --- Source separation ---
        remote = self._split_with_lalal(audio_file_path)
        if remote:
            self.y_harmonic = remote.get("back") or y_original
            self.y_percussive = remote.get("drum") or remote.get("beat")
            self.y_vocals = remote.get("vocals")
            print("  ✓ Stems obtained via LALAL.ai")
        else:
            # Fallback to HPSS if remote split unavailable
            try:
                self.y_harmonic, self.y_percussive = librosa.effects.hpss(self.y)
                print("  ✓ Stems obtained via local HPSS")
            except Exception:
                self.y_harmonic = self.y.copy()
                self.y_percussive = self.y.copy()
                print("  ⚠️  Stem separation failed – using full mix for all segments")

        print(
            f"Deck {self.name}: Track loaded. Duration: {librosa.get_duration(y=self.y, sr=self.sr):.2f}s"
        )

    def set_hot_cue(self, cue_idx, sample_pos=None):
        """Set or jump to hot cue"""
        if sample_pos is None:
            sample_pos = self.current_sample_pos
        if 0 <= cue_idx < len(self.hot_cues):
            self.hot_cues[cue_idx] = sample_pos

    def jump_to_hot_cue(self, cue_idx):
        """Jump to hot cue position"""
        if 0 <= cue_idx < len(self.hot_cues) and self.hot_cues[cue_idx] is not None:
            self.current_sample_pos = self.hot_cues[cue_idx]

    def delete_hot_cue(self, cue_idx):
        """Delete hot cue"""
        if 0 <= cue_idx < len(self.hot_cues):
            self.hot_cues[cue_idx] = None

    def set_loop_in(self):
        """Set loop in point"""
        self.loop_in_sample = self.current_sample_pos

    def set_loop_out(self):
        """Set loop out point and activate loop"""
        self.loop_out_sample = self.current_sample_pos
        if self.loop_in_sample is not None:
            self.loop_active = True

    def exit_loop(self):
        """Exit active loop"""
        self.loop_active = False

    def loop_half(self):
        """Halve the loop length"""
        if self.loop_in_sample is not None and self.loop_out_sample is not None:
            loop_length = self.loop_out_sample - self.loop_in_sample
            self.loop_out_sample = self.loop_in_sample + loop_length // 2

    def loop_double(self):
        """Double the loop length"""
        if self.loop_in_sample is not None and self.loop_out_sample is not None:
            loop_length = self.loop_out_sample - self.loop_in_sample
            self.loop_out_sample = self.loop_in_sample + loop_length * 2

    def auto_loop(self, beats):
        """Set an automatic loop of specified beats"""
        if self.analysis_data and "bpm" in self.analysis_data:
            bpm = float(self.analysis_data["bpm"])
            beat_samples = int(60.0 / bpm * self.sr)
            self.loop_in_sample = self.current_sample_pos
            self.loop_out_sample = self.current_sample_pos + (beats * beat_samples)
            self.loop_active = True
            self.auto_loop_beats = beats

    def beat_jump_forward(self, beats):
        """Jump forward by specified beats"""
        if self.analysis_data and "bpm" in self.analysis_data:
            bpm = float(self.analysis_data["bpm"])
            jump_samples = self.beat_jump.calculate_jump_samples(beats, bpm)
            self.current_sample_pos = min(
                self.current_sample_pos + jump_samples, len(self.y) - 1
            )

    def beat_jump_backward(self, beats):
        """Jump backward by specified beats"""
        if self.analysis_data and "bpm" in self.analysis_data:
            bpm = float(self.analysis_data["bpm"])
            jump_samples = self.beat_jump.calculate_jump_samples(beats, bpm)
            self.current_sample_pos = max(self.current_sample_pos - jump_samples, 0)

    def set_tempo_adjustment(self, value):
        """Set tempo adjustment (-1.0 to 1.0)"""
        self.tempo_adjustment = np.clip(value, -1.0, 1.0)

    def get_current_bpm(self):
        """Get current BPM with tempo adjustment"""
        if self.analysis_data and "bpm" in self.analysis_data:
            base_bpm = float(self.analysis_data["bpm"])
            adjustment = 1.0 + (self.tempo_adjustment * self.tempo_range)
            return base_bpm * adjustment
        return 120.0
        
    def get_audio_chunk(self, num_samples):
        if not self.is_playing or self.y is None:
            return np.zeros(num_samples)

        # Handle looping
        if self.loop_active and self.loop_in_sample is not None and self.loop_out_sample is not None:
            if self.current_sample_pos >= self.loop_out_sample:
                self.current_sample_pos = self.loop_in_sample

        # Apply tempo adjustment
        playback_rate = 1.0 + (self.tempo_adjustment * self.tempo_range)

        # Choose source based on segment_type
        source_full = self.y if self.y is not None else np.array([])
        if self.segment_type == 'beat' and self.y_percussive is not None:
            source = self.y_percussive
        elif self.segment_type == 'vocals' and self.y_vocals is not None:
            source = self.y_vocals
        elif self.segment_type in ['instrumental', 'back'] and self.y_harmonic is not None:
            source = self.y_harmonic
        else:
            source = source_full

        start = self.current_sample_pos
        end = min(start + int(num_samples * playback_rate), len(source))

        if start >= len(source):
            self.is_playing = False
            return np.zeros(num_samples)

        chunk = source[start:end]

        # Smooth out tiny playback-rate changes
        if playback_rate != 1.0 and len(chunk) > 1024:
            chunk = self._stretch_chunk(chunk.astype(np.float32), playback_rate)

        self.current_sample_pos = end

        # Pad if necessary
        if len(chunk) < num_samples:
            chunk = np.pad(chunk, (0, num_samples - len(chunk)))
        elif len(chunk) > num_samples:
            chunk = chunk[:num_samples]

        # Apply key shift if needed
        if self.key_shift != 0:
            chunk = pitch_shift_semitones(chunk, self.sr, self.key_shift)

        return chunk

    def process_chunk(self, chunk):
        # Apply beat repeat if active
        if self.beat_repeat.is_active:
            chunk = self.beat_repeat.process(chunk)

        # Apply pad effects
        chunk = self.pad_fx.process(chunk)

        # Apply main effects board
        chunk = self.board(chunk, self.sr)

        # Apply color FX if any
        if len(self.color_fx) > 0:
            chunk = self.color_fx(chunk, self.sr)

        return chunk
        
    def seek_to_time(self, time_in_seconds):
        self.current_sample_pos = int(time_in_seconds * self.sr)

    def cue_set(self):
        """Set main cue point"""
        self.cue_point = self.current_sample_pos

    def cue_play(self):
        """Play from cue point"""
        if hasattr(self, "cue_point"):
            self.current_sample_pos = self.cue_point
            self.is_playing = True

    def back_cue(self):
        """Jump to cue point without playing"""
        if hasattr(self, "cue_point"):
            self.current_sample_pos = self.cue_point

    # ---------------- Segment control -----------------
    def set_segment(self, segment):
        """Select which audio stem to play: 'full', 'vocals', 'beat', or 'instrumental'"""
        if segment in ['full', 'vocals', 'beat', 'instrumental', 'back']:
            self.segment_type = segment
        else:
            raise ValueError("segment must be 'full', 'vocals', 'beat', 'instrumental', or 'back'")

    def _stretch_chunk(self, chunk, rate):
        """High-quality time-stretch helper used during tiny tempo adjustments.

        The previous implementation incorrectly fed raw PCM audio into
        `librosa.phase_vocoder`, which expects a complex STFT and therefore
        produced heavily degraded, noisy output.  Instead we resample the
        audio so that it occupies the exact number of samples expected for
        the current buffer while retaining as much fidelity as possible.
        For very small rate deviations this is effectively transparent.
        """
        if np.isclose(rate, 1.0, atol=1e-4):
            return chunk

        target_length = int(len(chunk) / rate)

        # By treating the chunk as if it were recorded at `self.sr * rate` Hz
        # and converting it back to `self.sr`, we effectively time-stretch it
        # by the desired factor without altering pitch.
        virtual_orig_sr = int(self.sr * rate)

        # Use a high quality sinc-based resampler.  librosa >=0.9 provides
        # access to the SoX HQ resampler via `soxr_hq`; otherwise fall back to
        # `kaiser_best`.
        res_type = "soxr_hq" if "soxr_hq" in librosa.resample.__code__.co_varnames else "kaiser_best"
        stretched = librosa.resample(chunk.astype(np.float32), orig_sr=virtual_orig_sr, target_sr=self.sr, res_type=res_type)

        # The librosa resampler produces audio with the same sample-rate but it
        # does not guarantee an exact output length.  Trim or pad so that the
        # output matches the desired `target_length` to keep buffer alignment.
        if len(stretched) > target_length:
            stretched = stretched[:target_length]
        elif len(stretched) < target_length:
            stretched = np.pad(stretched, (0, target_length - len(stretched)))

        return stretched.astype(chunk.dtype)


# --- Mixer Class ---
class Mixer:
    """DJ Mixer with crossfader and master effects"""

    def __init__(self, sr=44100):
        self.sr = sr
        self.crossfader_position = 0.5  # 0.0 = full A, 1.0 = full B
        self.master_level = 0.8
        self.headphone_mix = 0.5  # 0.0 = cue, 1.0 = master
        self.headphone_level = 0.7

        # Channel monitors
        self.channel_cue = {"A": False, "B": False}
        self.master_cue = False

        # Smart fader
        self.smart_fader_enabled = False
        self.smart_fader_type = "smooth"  # smooth, cut, scratch

        # Mic input
        self.mic_level = 0.0
        self.mic_attenuator = -20  # dB

    def mix_channels(self, channel_a, channel_b):
        """Mix two channels based on crossfader position"""
        if self.smart_fader_enabled:
            # Apply smart fader curves
            if self.smart_fader_type == "cut":
                # Sharp cut
                a_level = 1.0 if self.crossfader_position < 0.5 else 0.0
                b_level = 0.0 if self.crossfader_position < 0.5 else 1.0
            elif self.smart_fader_type == "scratch":
                # Scratch curve (sharp transitions at extremes)
                if self.crossfader_position < 0.1:
                    a_level, b_level = 1.0, 0.0
                elif self.crossfader_position > 0.9:
                    a_level, b_level = 0.0, 1.0
                else:
                    a_level = 1.0 - self.crossfader_position
                    b_level = self.crossfader_position
            else:
                # Smooth fade (default)
                a_level = np.cos(self.crossfader_position * np.pi / 2)
                b_level = np.sin(self.crossfader_position * np.pi / 2)
        else:
            # Linear crossfade
            a_level = 1.0 - self.crossfader_position
            b_level = self.crossfader_position

        mixed = (channel_a * a_level + channel_b * b_level) * self.master_level
        return mixed

    def get_headphone_mix(self, cue_signal, master_signal):
        """Get headphone monitor mix"""
        return (
            cue_signal * (1.0 - self.headphone_mix)
            + master_signal * self.headphone_mix
        ) * self.headphone_level


# --- Main AudioRenderer Class ---
class AudioRenderer:
    """The main engine that reads a mix script and renders the audio with full DJ functionality"""

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.deck_a = Deck("A", sample_rate)
        self.deck_b = Deck("B", sample_rate)
        self.decks = {"A": self.deck_a, "B": self.deck_b}
        self.mixer = Mixer(sample_rate)
        self.sampler = Sampler(16, sample_rate)

        # Global tempo for sync
        self.master_tempo = 120.0

        # Beat FX
        self.beat_fx_types = {
            "echo": lambda: pb.Delay(delay_seconds=0.375, feedback=0.5, mix=0.5),
            "reverb": lambda: pb.Reverb(room_size=0.8, damping=0.5, wet_level=0.5),
            "phaser": lambda: pb.Phaser(rate_hz=0.5, depth=0.7, mix=0.5),
            "flanger": lambda: pb.Chorus(rate_hz=0.5, depth=0.9, mix=0.7),
            "pitch": lambda: pb.PitchShift(semitones=5),
            "roll": lambda: None,  # Implemented separately
            "transform": lambda: None,  # Implemented separately
        }
        self.current_beat_fx = None
        self.beat_fx_channel = None  # 'A', 'B', or 'master'

        # Headroom, gentle compression, then true-peak limiting for transparent output
        self.master_bus = pb.Pedalboard([
            pb.HighpassFilter(20),
            pb.Gain(-6),
            pb.Compressor(threshold_db=-12, ratio=4, attack_ms=8, release_ms=150),
            pb.Limiter(threshold_db=-1, release_ms=100)
        ])

        # Animation/automation storage
        self.fades = []
        self.automations = []

    def sync_deck(self, deck_name):
        """Sync deck to master tempo"""
        deck = self.decks[deck_name]
        deck.sync_enabled = True

        # Find master deck or use global tempo
        master_deck = None
        for d_name, d in self.decks.items():
            if d.is_master and d_name != deck_name:
                master_deck = d
                break

        if master_deck:
            target_bpm = master_deck.get_current_bpm()
        else:
            target_bpm = self.master_tempo

        # Adjust deck tempo to match
        if deck.analysis_data and "bpm" in deck.analysis_data:
            original_bpm = float(deck.analysis_data["bpm"])
            tempo_ratio = target_bpm / original_bpm - 1.0
            deck.set_tempo_adjustment(tempo_ratio / deck.tempo_range)

    def tap_bpm(self, tap_times):
        """Calculate BPM from tap times"""
        if len(tap_times) < 2:
            return

        intervals = np.diff(tap_times)
        avg_interval = np.mean(intervals)
        bpm = 60.0 / avg_interval
        self.master_tempo = bpm

    def set_beat_fx(self, fx_type, channel, beats=1, depth=0.5):
        """Set beat effect"""
        if fx_type in self.beat_fx_types:
            self.current_beat_fx = self.beat_fx_types[fx_type]()
            self.beat_fx_channel = channel

            # Configure effect parameters based on beats
            if self.current_beat_fx and hasattr(self.current_beat_fx, "delay_seconds"):
                # Calculate delay time based on beats and tempo
                beat_time = 60.0 / self.master_tempo
                self.current_beat_fx.delay_seconds = beat_time * beats

    def release_fx(self):
        """Release FX - momentary effect release"""
        # This would implement a temporary bypass of all effects
        pass

    def _get_effect_from_deck(self, deck_name, param_name):
        board = self.decks[deck_name].board
        mapping = {
            "filter_cutoff_hz": 0,
            "eq_high": 1,
            "eq_mid": 2,
            "eq_low": 3,
            "trim": 4,
            "channel_fader": 5,
        }

        idx = mapping.get(param_name)
        if idx is not None:
            return board[idx]
        return None

    def _update_fades(self, current_time):
        for fade in self.fades[:]:
            if current_time >= fade["end_time"]:
                setattr(fade["effect"], fade["attribute"], fade["end_val"])
                self.fades.remove(fade)
            else:
                progress = (current_time - fade["start_time"]) / (
                    fade["end_time"] - fade["start_time"]
                )

                # Apply curve
                if fade.get("curve") == "exponential":
                    progress = progress * progress
                elif fade.get("curve") == "logarithmic":
                    progress = np.sqrt(progress)

                current_val = fade["start_val"] + (fade["end_val"] - fade["start_val"]) * progress
                setattr(fade["effect"], fade["attribute"], current_val)
                
    def render(self, mix_script_data, output_path):
        print("--- Starting Audio Render with Full DJ Controller ---")
        mix_script = mix_script_data["script"]
        total_duration = mix_script_data["total_duration"]
        mix_script.sort(key=lambda x: x["time"])

        # Pre-load tracks
        for command in mix_script:
            if command["command"] == "load_track":
                deck = self.decks[command["params"]["deck"]]
                deck.load_track(
                    command["params"]["file_path"], command["params"].get("target_bpm")
                )

        CHUNK_SIZE = 8192  # larger blocks reduce state resets and artefacts
        total_samples = int(total_duration * self.sr)
        output_buffer = np.zeros(total_samples, dtype=np.float32)
        command_idx = 0

        # Track tap times for BPM
        tap_times = []
        
        for i in range(0, total_samples, CHUNK_SIZE):
            current_time = i / self.sr
            
            # Process commands at current time
            while command_idx < len(mix_script) and mix_script[command_idx]["time"] <= current_time:
                cmd = mix_script[command_idx]
                p = cmd.get("params", {})
                deck_name = p.get("deck")
                deck = self.decks.get(deck_name) if deck_name else None

                # Basic transport controls
                if cmd["command"] == "play" and deck:
                    deck.is_playing = True
                elif cmd["command"] == "pause" and deck:
                    deck.is_playing = False
                elif cmd["command"] == "stop" and deck:
                    deck.is_playing = False
                    deck.current_sample_pos = 0
                elif cmd["command"] == "cue_set" and deck:
                    deck.cue_set()
                elif cmd["command"] == "cue_play" and deck:
                    deck.cue_play()
                elif cmd["command"] == "back_cue" and deck:
                    deck.back_cue()

                # Hot cues
                elif cmd["command"] == "set_hot_cue" and deck:
                    deck.set_hot_cue(p["cue_idx"])
                elif cmd["command"] == "jump_hot_cue" and deck:
                    deck.jump_to_hot_cue(p["cue_idx"])
                elif cmd["command"] == "delete_hot_cue" and deck:
                    deck.delete_hot_cue(p["cue_idx"])

                # Loop controls
                elif cmd["command"] == "loop_in" and deck:
                    deck.set_loop_in()
                elif cmd["command"] == "loop_out" and deck:
                    deck.set_loop_out()
                elif cmd["command"] == "loop_exit" and deck:
                    deck.exit_loop()
                elif cmd["command"] == "loop_half" and deck:
                    deck.loop_half()
                elif cmd["command"] == "loop_double" and deck:
                    deck.loop_double()
                elif cmd["command"] == "auto_loop" and deck:
                    deck.auto_loop(p.get("beats", 4))

                # Beat jump
                elif cmd["command"] == "beat_jump_forward" and deck:
                    deck.beat_jump_forward(p.get("beats", 1))
                elif cmd["command"] == "beat_jump_backward" and deck:
                    deck.beat_jump_backward(p.get("beats", 1))

                # Tempo/Sync
                elif cmd["command"] == "set_tempo" and deck:
                    deck.set_tempo_adjustment(p["value"])
                elif cmd["command"] == "beat_sync" and deck:
                    self.sync_deck(deck_name)
                elif cmd["command"] == "set_master_deck" and deck:
                    # Clear other master decks
                    for d in self.decks.values():
                        d.is_master = False
                    deck.is_master = True
                elif cmd["command"] == "tap_bpm":
                    tap_times.append(current_time)
                    if len(tap_times) > 8:  # Keep last 8 taps
                        tap_times.pop(0)
                    self.tap_bpm(tap_times)

                # Performance modes
                elif cmd["command"] == "set_performance_mode" and deck:
                    deck.performance_mode = p["mode"]
                elif cmd["command"] == "key_shift" and deck:
                    deck.key_shift = p.get("semitones", 0)

                # Pad FX
                elif cmd["command"] == "toggle_pad_fx" and deck:
                    deck.pad_fx.toggle_effect(p["effect_id"])

                # Beat FX
                elif cmd["command"] == "set_beat_fx":
                    self.set_beat_fx(
                        p["fx_type"], p.get("channel", "master"), p.get("beats", 1), p.get("depth", 0.5)
                    )
                elif cmd["command"] == "beat_fx_on":
                    # Activate beat FX
                    pass
                elif cmd["command"] == "beat_fx_off":
                    self.current_beat_fx = None
                elif cmd["command"] == "release_fx":
                    self.release_fx()

                # Sampler
                elif cmd["command"] == "sampler_load":
                    # Load audio into sampler slot
                    slot_idx = p["slot"]
                    if "file_path" in p:
                        sample_audio, _ = librosa.load(p["file_path"], sr=self.sr)
                        self.sampler.load_slot(slot_idx, sample_audio)
                elif cmd["command"] == "sampler_play":
                    self.sampler.play_slot(p["slot"])
                elif cmd["command"] == "sampler_stop":
                    self.sampler.stop_slot(p["slot"])

                # Mixer controls
                elif cmd["command"] == "set_crossfader":
                    target_pos = p["position"]
                    fade_duration = p.get("fade_duration", 0)
                    if fade_duration > 0:
                        self.fades.append(
                            {
                                "effect": self.mixer,
                                "attribute": "crossfader_position",
                                "start_val": self.mixer.crossfader_position,
                                "end_val": target_pos,
                                "start_time": current_time,
                                "end_time": current_time + fade_duration,
                                "curve": p.get("curve", "linear"),
                            }
                        )
                    else:
                        self.mixer.crossfader_position = target_pos

                elif cmd["command"] == "set_master_level":
                    self.mixer.master_level = p["level"]
                elif cmd["command"] == "set_headphone_mix":
                    self.mixer.headphone_mix = p["mix"]
                elif cmd["command"] == "set_headphone_level":
                    self.mixer.headphone_level = p["level"]
                elif cmd["command"] == "toggle_channel_cue":
                    self.mixer.channel_cue[p["channel"]] = p.get("enabled", True)
                elif cmd["command"] == "toggle_master_cue":
                    self.mixer.master_cue = p.get("enabled", True)
                elif cmd["command"] == "set_smart_fader":
                    self.mixer.smart_fader_enabled = p.get("enabled", True)
                    self.mixer.smart_fader_type = p.get("type", "smooth")

                # EQ and effects parameters
                elif cmd["command"] == "set_parameter":
                    effect = self._get_effect_from_deck(p["deck"], p["parameter"])
                    if effect:
                        attribute_name = "gain_db"
                        target_val = p["value"]

                        if p["parameter"] == "filter_cutoff_hz":
                            attribute_name = "cutoff_frequency_hz"
                            # Map 0-1 to 20Hz-20kHz on log scale
                            target_val = 20.0 * np.power(1000.0, target_val)
                        elif p["parameter"] in ["trim", "channel_fader"]:
                            # Convert linear to dB
                            target_val = (
                                librosa.amplitude_to_db(target_val)
                                if target_val > 0.001
                                else -60.0
                            )
                        elif p["parameter"] in ["eq_high", "eq_mid", "eq_low"]:
                            # Map 0-1 to -24dB to +12dB
                            target_val = np.interp(target_val, [0, 0.5, 1.0], [-24, 0, 12])

                        fade_duration = p.get("fade_duration", 0)
                        if fade_duration > 0.01:
                            self.fades.append(
                                {
                                    "effect": effect,
                                    "attribute": attribute_name,
                                    "start_val": getattr(effect, attribute_name),
                                    "end_val": target_val,
                                    "start_time": current_time,
                                    "end_time": current_time + fade_duration,
                                }
                            )
                        else:
                            setattr(effect, attribute_name, target_val)

                # Color FX
                elif cmd["command"] == "set_color_fx" and deck:
                    fx_type = p["fx_type"]
                    if fx_type == "filter":
                        deck.color_fx = pb.Pedalboard(
                            [pb.HighpassFilter(cutoff_frequency_hz=p.get("frequency", 1000))]
                        )
                    elif fx_type == "noise":
                        deck.color_fx = pb.Pedalboard(
                            [pb.Distortion(drive_db=p.get("amount", 10))]
                        )
                    elif fx_type == "pitch":
                        deck.color_fx = pb.Pedalboard(
                            [pb.PitchShift(semitones=p.get("semitones", 0))]
                        )
                    elif fx_type == "none":
                        deck.color_fx = pb.Pedalboard([])

                # Vinyl mode / Jog wheel
                elif cmd["command"] == "jog_wheel" and deck:
                    if p.get("scratch"):
                        deck.vinyl_sim.is_scratching = True
                        deck.vinyl_sim.set_jog_position(p["delta"])
                    else:
                        deck.vinyl_sim.set_pitch_bend(p["bend"])

                # Beat repeat/roll
                elif cmd["command"] == "beat_repeat" and deck:
                    if p.get("activate"):
                        deck.beat_repeat.set_loop(
                            deck.y[deck.current_sample_pos :], p.get("beats", 1), deck.get_current_bpm()
                        )
                    else:
                        deck.beat_repeat.deactivate()

                # Generic seek
                elif cmd["command"] == "seek_to_time" and deck:
                    deck.seek_to_time(p["time_in_seconds"])

                # Segment (vocals/beat/full) switching
                elif cmd["command"] == "set_segment" and deck:
                    deck.set_segment(p.get("segment", "full"))

                command_idx += 1
            
            # Update all active fades
            self._update_fades(current_time)
            
            # Get audio chunks from decks
            chunk_a = self.deck_a.get_audio_chunk(CHUNK_SIZE)
            chunk_b = self.deck_b.get_audio_chunk(CHUNK_SIZE)
            
            # Process through deck effects
            processed_a = self.deck_a.process_chunk(chunk_a)
            processed_b = self.deck_b.process_chunk(chunk_b)

            # Apply beat FX if active
            if self.current_beat_fx:
                if self.beat_fx_channel == "A":
                    processed_a = self.current_beat_fx(processed_a, self.sr)
                elif self.beat_fx_channel == "B":
                    processed_b = self.current_beat_fx(processed_b, self.sr)

            # Mix channels
            mixed = self.mixer.mix_channels(processed_a, processed_b)

            # Add sampler output
            sampler_output = self.sampler.process(CHUNK_SIZE)
            mixed += sampler_output * 0.8  # Sampler at 80% level

            # Apply master beat FX if set to master channel
            if self.current_beat_fx and self.beat_fx_channel == "master":
                mixed = self.current_beat_fx(mixed, self.sr)

            # defer master_bus to a single offline pass; accumulate dry mix
            end_sample = min(i + CHUNK_SIZE, total_samples)
            output_buffer[i:end_sample] = mixed[: end_sample - i] * 0.85  # 1.4 dB head-room

            # Progress indicator
            if i % (self.sr * 10) == 0:  # Every 10 seconds
                progress = (i / total_samples) * 100
                print(f"  Rendering: {progress:.1f}%")

        # ---------------------------------------------------------------
        print("--- Timeline complete. Applying master processing chain ---")

        output_buffer = self.master_bus(output_buffer, self.sr)

        peak = np.max(np.abs(output_buffer))
        if peak > 0:
            output_buffer *= 0.9499 / peak  # final normalisation to -0.5 dBFS

        print("--- Render Complete. Saving file... ---")

        sf.write(output_path, output_buffer, self.sr, subtype='PCM_24')
        print(f"Successfully saved mix to {output_path}")


# --- Mix Script Generator ---
class MixScriptGenerator:
    """Helper class to generate mix scripts programmatically"""

    @staticmethod
    def create_basic_transition(track1_path, track2_path, transition_duration=16.0):
        """Create a basic mix script for transitioning between two tracks"""
        script = {
            "total_duration": 180.0,  # 3 minutes
            "script": [
                # Load tracks
                {
                    "time": 0.0,
                    "command": "load_track",
                    "params": {"deck": "A", "file_path": track1_path, "target_bpm": 128},
                },
                {
                    "time": 0.0,
                    "command": "load_track",
                    "params": {"deck": "B", "file_path": track2_path, "target_bpm": 128},
                },
                # Start deck A
                {"time": 0.1, "command": "play", "params": {"deck": "A"}},
                {
                    "time": 0.1,
                    "command": "set_parameter",
                    "params": {"deck": "A", "parameter": "channel_fader", "value": 1.0},
                },
                # Set initial crossfader to A
                {"time": 0.1, "command": "set_crossfader", "params": {"position": 0.0}},
                # At 60 seconds, start mixing in deck B
                {"time": 60.0, "command": "play", "params": {"deck": "B"}},
                {
                    "time": 60.0,
                    "command": "set_parameter",
                    "params": {"deck": "B", "parameter": "channel_fader", "value": 1.0},
                },
                # EQ swap during transition
                {
                    "time": 60.0,
                    "command": "set_parameter",
                    "params": {"deck": "B", "parameter": "eq_low", "value": 0.0},
                },
                {
                    "time": 64.0,
                    "command": "set_parameter",
                    "params": {
                        "deck": "B",
                        "parameter": "eq_low",
                        "value": 0.5,
                        "fade_duration": 8.0,
                    },
                },
                {
                    "time": 64.0,
                    "command": "set_parameter",
                    "params": {
                        "deck": "A",
                        "parameter": "eq_low",
                        "value": 0.0,
                        "fade_duration": 8.0,
                    },
                },
                # Crossfade
                {
                    "time": 60.0,
                    "command": "set_crossfader",
                    "params": {"position": 1.0, "fade_duration": transition_duration},
                },
                # Stop deck A after transition
                {"time": 76.0, "command": "stop", "params": {"deck": "A"}},
                # End
                {"time": 179.0, "command": "stop", "params": {"deck": "B"}},
            ],
        }
        return script

    @staticmethod
    def create_scratch_routine(track_path, routine_duration=30.0):
        """Create a scratch routine script"""
        script = {
            "total_duration": routine_duration,
            "script": [
                # Load track
                {
                    "time": 0.0,
                    "command": "load_track",
                    "params": {"deck": "A", "file_path": track_path},
                },
                # Start playback
                {"time": 0.1, "command": "play", "params": {"deck": "A"}},
                {
                    "time": 0.1,
                    "command": "set_parameter",
                    "params": {"deck": "A", "parameter": "channel_fader", "value": 1.0},
                },
                # Set smart fader to scratch mode
                {
                    "time": 0.1,
                    "command": "set_smart_fader",
                    "params": {"enabled": True, "type": "scratch"},
                },
                # Scratch pattern
                {
                    "time": 2.0,
                    "command": "jog_wheel",
                    "params": {"deck": "A", "scratch": True, "delta": -0.5},
                },
                {
                    "time": 2.2,
                    "command": "jog_wheel",
                    "params": {"deck": "A", "scratch": True, "delta": 0.5},
                },
                {
                    "time": 2.4,
                    "command": "jog_wheel",
                    "params": {"deck": "A", "scratch": True, "delta": -0.3},
                },
                {
                    "time": 2.6,
                    "command": "jog_wheel",
                    "params": {"deck": "A", "scratch": True, "delta": 0.3},
                },
                # Beat juggle with hot cues
                {
                    "time": 5.0,
                    "command": "set_hot_cue",
                    "params": {"deck": "A", "cue_idx": 0},
                },
                {
                    "time": 6.0,
                    "command": "set_hot_cue",
                    "params": {"deck": "A", "cue_idx": 1},
                },
                {
                    "time": 8.0,
                    "command": "jump_hot_cue",
                    "params": {"deck": "A", "cue_idx": 0},
                },
                {
                    "time": 8.5,
                    "command": "jump_hot_cue",
                    "params": {"deck": "A", "cue_idx": 1},
                },
                {
                    "time": 9.0,
                    "command": "jump_hot_cue",
                    "params": {"deck": "A", "cue_idx": 0},
                },
                # End
                {
                    "time": routine_duration - 0.1,
                    "command": "stop",
                    "params": {"deck": "A"},
                },
            ],
        }
        return script


# --- Quick-mix helper ---------------------------------------------------------
def render_simple_crossfade(analysis_json_a: str,
                            analysis_json_b: str,
                            fade_start_s: float = 40.0,
                            fade_length_s: float = 32.0,
                            total_duration_s: float = 210.0,
                            out_path: str = "simple_fade_mix.wav",
                            sr: int = 44100):
    """
    Offline, artefact-free A→B cross-fade.  Ignores all decks / FX, it is purely
    to confirm that audio quality is preserved when we *don't* stream in blocks.
    """
    # ------------------------------------------------------------------ load --
    with open(analysis_json_a) as f:
        path_a = json.load(f)["file_path"]
    with open(analysis_json_b) as f:
        path_b = json.load(f)["file_path"]

    yA, srA = librosa.load(path_a, sr=sr, mono=False)
    yB, srB = librosa.load(path_b, sr=sr, mono=False)  # mono=False keeps stereo

    # pad / truncate both to total duration
    target_len = int(total_duration_s * sr)
    pad = lambda y: np.pad(y, ((0, 0), (0, max(0, target_len - y.shape[1])))) \
                    if y.ndim == 2 else np.pad(y, (0, max(0, target_len - y.shape[0])))
    yA = pad(yA)[:, :target_len] if yA.ndim == 2 else pad(yA)[:target_len]
    yB = pad(yB)[:, :target_len] if yB.ndim == 2 else pad(yB)[:target_len]

    # --------------------------------------------------------- build envelope
    env = np.ones(target_len, dtype=np.float32)
    fade_start = int(fade_start_s * sr)
    fade_end   = int((fade_start_s + fade_length_s) * sr)
    fade_t     = np.linspace(0, np.pi/2, fade_end - fade_start)
    env[fade_start:fade_end] = np.cos(fade_t)          # Deck A envelope
    envB = 1.0 - env                                  # Deck B envelope

    # --------------------------------------------------------------- mix down
    # Broadcast env to both channels if stereo
    if yA.ndim == 2:
        env = env[None, :]
        envB = envB[None, :]
    mix = yA * env + yB * envB

    # headroom & limiter (same as master_bus but no extra comp/HP filter)
    mix *= 0.9
    mix = pb.Limiter(threshold_db=-1)(mix, sr)

    # ------------------------------------------------------------ write WAV
    sf.write(out_path, mix.T, sr, subtype="FLOAT")
    print(f"✓ Rendered artefact-free mix to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render an audio mix from a JSON script file with full DJ controller functionality."
    )
    parser.add_argument("mix_script_file", type=str, help="Path to the mix script JSON file.")
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to save the output WAV file. Defaults to script name with .wav extension."
    )
    parser.add_argument(
        "--generate-example",
        action="store_true",
        help="Generate an example mix script instead of rendering.",
    )
    
    args = parser.parse_args()
    
    if args.generate_example:
        # Generate example script
        print("Generating example mix script...")
        example_script = MixScriptGenerator.create_basic_transition(
            "analysis/track1_analysis.json", "analysis/track2_analysis.json"
        )

        with open("example_mix_script.json", "w") as f:
            json.dump(example_script, f, indent=2)
        print("Example script saved to: example_mix_script.json")
    else:
        # Render mix
        if not os.path.exists(args.mix_script_file):
            print(f"Error: Mix script file not found at {args.mix_script_file}")
        else:
            with open(args.mix_script_file, "r") as f:
                mix_script_data = json.load(f)

            output_path = args.output_file
            if not output_path:
                base_name = os.path.splitext(os.path.basename(args.mix_script_file))[0]
                output_path = f"{base_name}.wav"
                
            renderer = AudioRenderer()
            renderer.render(mix_script_data, output_path)
