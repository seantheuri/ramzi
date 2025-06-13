import librosa
import numpy as np
import soundfile as sf
import json
import os
import argparse
import pedalboard as pb  # Import pedalboard

# --- Helper Function for Time-Stretching (Unchanged) ---
def stretch_audio_to_bpm(y, sr, original_bpm, target_bpm):
    if np.isclose(original_bpm, target_bpm):
        return y
    stretch_rate = target_bpm / original_bpm
    print(f"  - Stretching from {original_bpm:.2f} BPM to {target_bpm:.2f} BPM (rate: {stretch_rate:.3f})")
    return librosa.effects.time_stretch(y=y, rate=stretch_rate)


# --- Core Component Classes (Refactored for pedalboard) ---

class Deck:
    """Represents a single virtual DJ deck with its own effects board."""
    def __init__(self, name, sample_rate=44100):
        self.name = name
        self.y = None
        self.sr = sample_rate
        self.analysis_data = None
        self.is_playing = False
        self.current_sample_pos = 0
        self.board = pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=20.0), # Index 0: Filter
            pb.HighShelfFilter(cutoff_frequency_hz=4000, gain_db=0), # Index 1: EQ High
            pb.PeakFilter(cutoff_frequency_hz=1000, q=1.0, gain_db=0), # Index 2: EQ Mid
            pb.LowShelfFilter(cutoff_frequency_hz=250, gain_db=0),   # Index 3: EQ Low
            pb.Gain(gain_db=0) # Index 4: Volume control
        ])

    def load_track(self, analysis_file_path, target_bpm):
        print(f"Deck {self.name}: Loading track from {analysis_file_path}")
        with open(analysis_file_path, 'r') as f: self.analysis_data = json.load(f)
        audio_file_path = self.analysis_data['file_path']
        if not os.path.exists(audio_file_path): raise FileNotFoundError(f"Audio file '{audio_file_path}' from analysis json not found.")
        y_original, sr_original = librosa.load(audio_file_path, sr=None)
        if sr_original != self.sr: y_original = librosa.resample(y=y_original, orig_sr=sr_original, target_sr=self.sr)
        original_bpm = float(self.analysis_data['bpm'])
        self.y = stretch_audio_to_bpm(y_original, self.sr, original_bpm, target_bpm)
        print(f"Deck {self.name}: Track loaded. Duration: {librosa.get_duration(y=self.y, sr=self.sr):.2f}s")
        
    def get_audio_chunk(self, num_samples):
        if not self.is_playing or self.y is None: return np.zeros(num_samples)
        start, end = self.current_sample_pos, self.current_sample_pos + num_samples
        if start >= len(self.y): self.is_playing = False; return np.zeros(num_samples)
        chunk = self.y[start:end]; self.current_sample_pos = end
        if len(chunk) < num_samples: chunk = np.pad(chunk, (0, num_samples - len(chunk)))
        return chunk

    def process_chunk(self, chunk):
        return self.board(chunk, self.sr)
        
    def seek_to_time(self, time_in_seconds):
        self.current_sample_pos = int(time_in_seconds * self.sr)


# --- AudioRenderer Class (Updated with Limiter and Seek) ---
class AudioRenderer:
    """The main engine that reads a mix script and renders the audio."""
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.deck_a = Deck('A', sample_rate)
        self.deck_b = Deck('B', sample_rate)
        self.decks = {'A': self.deck_a, 'B': self.deck_b}
        self.fades = []
        self.master_bus = pb.Pedalboard([pb.Limiter(threshold_db=-0.5, release_ms=100)])

    def _get_effect_from_deck(self, deck_name, param_name):
        board = self.decks[deck_name].board
        if param_name == 'filter_cutoff_hz': return board[0]
        if param_name == 'eq_high': return board[1]
        if param_name == 'eq_mid': return board[2]
        if param_name == 'eq_low': return board[3]
        if param_name == 'volume': return board[4]
        return None

    def _update_fades(self, current_time):
        for fade in self.fades[:]:
            if current_time >= fade['end_time']:
                setattr(fade['effect'], fade['attribute'], fade['end_val'])
                self.fades.remove(fade)
            else:
                progress = (current_time - fade['start_time']) / (fade['end_time'] - fade['start_time'])
                current_val = fade['start_val'] + (fade['end_val'] - fade['start_val']) * progress
                setattr(fade['effect'], fade['attribute'], current_val)
                
    def render(self, mix_script_data, output_path):
        print("--- Starting Audio Render from Script ---")
        mix_script = mix_script_data['script']
        total_duration = mix_script_data['total_duration']
        mix_script.sort(key=lambda x: x['time'])
        
        for command in mix_script:
            if command['command'] == 'load_track':
                deck = self.decks[command['params']['deck']]
                deck.load_track(command['params']['file_path'], command['params']['target_bpm'])
        
        CHUNK_SIZE, total_samples = 1024, int(total_duration * self.sr)
        output_buffer = np.zeros(total_samples, dtype=np.float32)
        command_idx = 0
        
        for i in range(0, total_samples, CHUNK_SIZE):
            current_time = i / self.sr
            
            while command_idx < len(mix_script) and mix_script[command_idx]['time'] <= current_time:
                cmd = mix_script[command_idx]
                p = cmd.get('params', {})
                deck = self.decks.get(p.get('deck'))
                if cmd['command'] == 'play' and deck: deck.is_playing = True
                elif cmd['command'] == 'stop' and deck: deck.is_playing = False
                elif cmd['command'] == 'seek_to_time' and deck: deck.seek_to_time(p['time_in_seconds'])
                elif cmd['command'] == 'set_parameter':
                    effect = self._get_effect_from_deck(p['deck'], p['parameter'])
                    if effect:
                        attribute_name, target_val = 'gain_db', p['value']
                        if p['parameter'] == 'filter_cutoff_hz': attribute_name = 'cutoff_frequency_hz'
                        elif p['parameter'] == 'volume': target_val = librosa.amplitude_to_db(target_val) if target_val > 0.001 else -100.0
                        else: target_val = np.interp(target_val, [0, 0.5, 1.0], [-24, 0, 6])
                        
                        fade_duration = p.get('fade_duration', 0)
                        if fade_duration > 0.01:
                            self.fades.append({'effect': effect, 'attribute': attribute_name, 'start_val': getattr(effect, attribute_name), 'end_val': target_val, 'start_time': current_time, 'end_time': current_time + fade_duration})
                        else: setattr(effect, attribute_name, target_val)
                command_idx += 1
            
            self._update_fades(current_time)
            
            chunk_a = self.deck_a.get_audio_chunk(CHUNK_SIZE)
            chunk_b = self.deck_b.get_audio_chunk(CHUNK_SIZE)
            
            processed_a = self.deck_a.process_chunk(chunk_a)
            processed_b = self.deck_b.process_chunk(chunk_b)
            mixed_chunk = processed_a + processed_b
            mastered_chunk = self.master_bus(mixed_chunk, self.sr)
            
            end_sample = min(i + CHUNK_SIZE, total_samples)
            output_buffer[i:end_sample] = mastered_chunk[:end_sample - i]

        print("--- Render Complete. Saving file... ---")
        sf.write(output_path, output_buffer, self.sr)
        print(f"Successfully saved mix to {output_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an audio mix from a JSON script file.")
    parser.add_argument("mix_script_file", type=str, help="Path to the mix script JSON file.")
    parser.add_argument("-o", "--output_file", type=str, help="Path to save the output WAV file. Defaults to script name with .wav extension.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mix_script_file):
        print(f"Error: Mix script file not found at {args.mix_script_file}")
    else:
        with open(args.mix_script_file, 'r') as f:
            mix_script_data = json.load(f)

        output_path = args.output_file
        if not output_path:
            base_name = os.path.splitext(os.path.basename(args.mix_script_file))[0]
            output_path = f"{base_name}.wav"
            
        renderer = AudioRenderer()
        renderer.render(mix_script_data, output_path)
