import os
import json
import tempfile
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Import your existing modules
# Make sure these are in the same directory or properly installed
try:
    from analysis import analyze_song
    from llm_dj import generate_mix_script
    from controller import AudioRenderer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure analysis.py, llm_dj.py, and controller.py are in the same directory")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['STATIC_FOLDER'] = 'static'

# Create necessary directories
for folder in ['uploads', 'outputs', 'static', 'temp']:
    os.makedirs(folder, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store job progress
job_progress = {}

class MixJob:
    """Represents a mix creation job with progress tracking"""
    def __init__(self, job_id):
        self.job_id = job_id
        self.status = 'pending'
        self.progress = 0
        self.current_step = ''
        self.error = None
        self.result = {}
        self.created_at = time.time()
        
    def update(self, status=None, progress=None, current_step=None, error=None):
        if status:
            self.status = status
        if progress is not None:
            self.progress = progress
        if current_step:
            self.current_step = current_step
        if error:
            self.error = error
            self.status = 'error'

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'mp3', 'wav', 'flac', 'm4a', 'ogg', 'aac'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def cleanup_old_files():
    """Clean up files older than 1 hour"""
    current_time = time.time()
    for folder in ['uploads', 'outputs', 'temp']:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                if current_time - os.path.getmtime(filepath) > 3600:  # 1 hour
                    try:
                        os.remove(filepath)
                    except:
                        pass

def process_mix_job(job_id, trackA_path, trackB_path, mix_style, user_prompt, target_bpm=None):
    """Process the mix creation in a separate thread"""
    job = job_progress[job_id]
    temp_dir = None
    
    try:
        # Create temporary directory for this job
        temp_dir = os.path.join('temp', job_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Step 1: Analyze Track A
        job.update(status='processing', progress=10, current_step='Analyzing Track A')
        logger.info(f"Job {job_id}: Analyzing track A")
        
        analysis_a_path = os.path.join(temp_dir, 'trackA_analysis.json')
        analysis_a = analyze_song(trackA_path, include_lyrics=False)  # Skip lyrics for speed
        if not analysis_a:
            raise Exception("Failed to analyze Track A")
        
        with open(analysis_a_path, 'w') as f:
            json.dump(analysis_a, f, indent=2)
        
        # Step 2: Analyze Track B
        job.update(progress=30, current_step='Analyzing Track B')
        logger.info(f"Job {job_id}: Analyzing track B")
        
        analysis_b_path = os.path.join(temp_dir, 'trackB_analysis.json')
        analysis_b = analyze_song(trackB_path, include_lyrics=False)
        if not analysis_b:
            raise Exception("Failed to analyze Track B")
            
        with open(analysis_b_path, 'w') as f:
            json.dump(analysis_b, f, indent=2)
        
        # Step 3: Generate mix script
        job.update(progress=50, current_step='Creating mix strategy with AI')
        logger.info(f"Job {job_id}: Generating mix script")
        
        mix_script_path = os.path.join(temp_dir, 'mix_script.json')
        
        # Call your LLM script generator
        from llm_dj import generate_mix_script as llm_generate_mix
        llm_generate_mix(
            analysis_a_path,
            analysis_b_path,
            user_prompt or f"Create a {mix_style} style mix",
            mix_script_path,
            mix_style=mix_style
        )
        
        # Step 4: Render the mix
        job.update(progress=70, current_step='Rendering audio mix')
        logger.info(f"Job {job_id}: Rendering mix")
        
        output_filename = f'mix_{job_id}.wav'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Load mix script
        with open(mix_script_path, 'r') as f:
            mix_script_data = json.load(f)
        
        # If target BPM specified, update the script
        if target_bpm:
            for cmd in mix_script_data['script']:
                if cmd['command'] == 'load_track':
                    cmd['params']['target_bpm'] = float(target_bpm)
        
        # Render using your controller
        from controller import AudioRenderer
        renderer = AudioRenderer()
        renderer.render(mix_script_data, output_path)
        
        # Step 5: Generate analysis visualization
        job.update(progress=90, current_step='Generating analysis visualization')
        logger.info(f"Job {job_id}: Creating visualization")
        
        # Generate waveform images and beat analysis
        visualization_data = create_mix_visualization(
            trackA_path, trackB_path, output_path,
            analysis_a, analysis_b, mix_script_data,
            temp_dir
        )
        
        # Move files to static folder for serving
        static_dir = os.path.join(app.config['STATIC_FOLDER'], job_id)
        os.makedirs(static_dir, exist_ok=True)
        
        # Copy audio file
        static_audio_path = os.path.join(static_dir, output_filename)
        shutil.copy2(output_path, static_audio_path)
        
        # Copy visualization files
        for key, filepath in visualization_data.items():
            if os.path.exists(filepath):
                filename = os.path.basename(filepath)
                static_path = os.path.join(static_dir, filename)
                shutil.copy2(filepath, static_path)
                visualization_data[key] = f'/static/{job_id}/{filename}'
        
        # Prepare result
        job.result = {
            'audioUrl': f'/static/{job_id}/{output_filename}',
            'scriptUrl': f'/api/download-script/{job_id}',
            'description': mix_script_data.get('description', ''),
            'techniques': mix_script_data.get('technique_highlights', []),
            'visualization': visualization_data,
            'analysis': {
                'trackA': {
                    'name': os.path.basename(trackA_path),
                    'bpm': analysis_a.get('bpm'),
                    'key': analysis_a.get('key_camelot'),
                    'energy': analysis_a.get('energy_normalized')
                },
                'trackB': {
                    'name': os.path.basename(trackB_path),
                    'bpm': analysis_b.get('bpm'),
                    'key': analysis_b.get('key_camelot'),
                    'energy': analysis_b.get('energy_normalized')
                },
                'mix': {
                    'duration': mix_script_data.get('total_duration', 0),
                    'target_bpm': target_bpm or 'auto',
                    'commands_count': len(mix_script_data.get('script', []))
                }
            }
        }
        
        job.update(status='completed', progress=100, current_step='Mix complete!')
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Error - {str(e)}")
        job.update(status='error', error=str(e))
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def create_mix_visualization(trackA_path, trackB_path, mix_path,
                           analysis_a, analysis_b, mix_script_data, output_dir):
    """Generate visualization data for the mix"""
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    import librosa.display
    
    visualization_data = {}
    
    try:
        # Load audio files
        y_a, sr_a = librosa.load(trackA_path, sr=22050, mono=True)
        y_b, sr_b = librosa.load(trackB_path, sr=22050, mono=True)
        y_mix, sr_mix = librosa.load(mix_path, sr=22050, mono=True)
        
        # 1. Waveform comparison
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle('Waveform Comparison', fontsize=16)
        
        # Track A waveform
        axes[0].set_title(f'Track A: {os.path.basename(trackA_path)}')
        librosa.display.waveshow(y_a, sr=sr_a, ax=axes[0], alpha=0.8)
        axes[0].set_ylabel('Amplitude')
        
        # Track B waveform
        axes[1].set_title(f'Track B: {os.path.basename(trackB_path)}')
        librosa.display.waveshow(y_b, sr=sr_b, ax=axes[1], alpha=0.8, color='orange')
        axes[1].set_ylabel('Amplitude')
        
        # Mix waveform
        axes[2].set_title('Final Mix')
        librosa.display.waveshow(y_mix, sr=sr_mix, ax=axes[2], alpha=0.8, color='green')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_xlabel('Time (s)')
        
        plt.tight_layout()
        waveform_path = os.path.join(output_dir, 'waveforms.png')
        plt.savefig(waveform_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualization_data['waveforms'] = waveform_path
        
        # 2. Beat alignment visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Extract tempo and beats
        tempo_a, beats_a = librosa.beat.beat_track(y=y_a, sr=sr_a)
        tempo_b, beats_b = librosa.beat.beat_track(y=y_b, sr=sr_b)
        tempo_mix, beats_mix = librosa.beat.beat_track(y=y_mix, sr=sr_mix)
        
        # Convert to time
        beat_times_a = librosa.frames_to_time(beats_a, sr=sr_a)
        beat_times_b = librosa.frames_to_time(beats_b, sr=sr_b)
        beat_times_mix = librosa.frames_to_time(beats_mix, sr=sr_mix)
        
        # Plot beat positions
        ax.vlines(beat_times_a[:100], 0, 1, alpha=0.5, color='blue', label=f'Track A ({tempo_a:.1f} BPM)')
        ax.vlines(beat_times_b[:100], 1, 2, alpha=0.5, color='orange', label=f'Track B ({tempo_b:.1f} BPM)')
        ax.vlines(beat_times_mix[:200], 2, 3, alpha=0.5, color='green', label=f'Mix ({tempo_mix:.1f} BPM)')
        
        ax.set_ylim(0, 3.5)
        ax.set_xlim(0, 30)  # First 30 seconds
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Track')
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['Track A', 'Track B', 'Mix'])
        ax.set_title('Beat Grid Alignment (First 30 seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        beat_path = os.path.join(output_dir, 'beat_alignment.png')
        plt.savefig(beat_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualization_data['beat_alignment'] = beat_path
        
        # 3. Energy comparison over time
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate RMS energy
        hop_length = 512
        rms_a = librosa.feature.rms(y=y_a, hop_length=hop_length)[0]
        rms_b = librosa.feature.rms(y=y_b, hop_length=hop_length)[0]
        rms_mix = librosa.feature.rms(y=y_mix, hop_length=hop_length)[0]
        
        # Convert to time
        frames_a = range(len(rms_a))
        frames_b = range(len(rms_b))
        frames_mix = range(len(rms_mix))
        
        t_a = librosa.frames_to_time(frames_a, sr=sr_a, hop_length=hop_length)
        t_b = librosa.frames_to_time(frames_b, sr=sr_b, hop_length=hop_length)
        t_mix = librosa.frames_to_time(frames_mix, sr=sr_mix, hop_length=hop_length)
        
        # Smooth the energy curves
        from scipy.ndimage import gaussian_filter1d
        rms_a_smooth = gaussian_filter1d(rms_a, sigma=10)
        rms_b_smooth = gaussian_filter1d(rms_b, sigma=10)
        rms_mix_smooth = gaussian_filter1d(rms_mix, sigma=10)
        
        ax.plot(t_a, rms_a_smooth, label='Track A', alpha=0.7, linewidth=2)
        ax.plot(t_b, rms_b_smooth, label='Track B', alpha=0.7, linewidth=2)
        ax.plot(t_mix, rms_mix_smooth, label='Mix', alpha=0.8, linewidth=2.5, color='green')
        
        # Mark mix points from script
        mix_events = []
        for cmd in mix_script_data.get('script', []):
            if cmd['command'] in ['play', 'stop', 'set_crossfader']:
                mix_events.append({
                    'time': cmd['time'],
                    'command': cmd['command'],
                    'deck': cmd.get('params', {}).get('deck', 'master')
                })
        
        for event in mix_events:
            ax.axvline(x=event['time'], color='red', linestyle='--', alpha=0.3)
            ax.text(event['time'], ax.get_ylim()[1]*0.9, 
                   f"{event['command']}\n{event['deck']}", 
                   rotation=90, verticalalignment='bottom', fontsize=8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (RMS)')
        ax.set_title('Energy Levels Throughout Mix')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        energy_path = os.path.join(output_dir, 'energy_comparison.png')
        plt.savefig(energy_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualization_data['energy_comparison'] = energy_path
        
        # 4. Spectral analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Spectral Analysis', fontsize=16)
        
        # Compute spectrograms
        D_a = librosa.amplitude_to_db(np.abs(librosa.stft(y_a)), ref=np.max)
        D_b = librosa.amplitude_to_db(np.abs(librosa.stft(y_b)), ref=np.max)
        D_mix = librosa.amplitude_to_db(np.abs(librosa.stft(y_mix)), ref=np.max)
        
        # Track A spectrogram
        img_a = librosa.display.specshow(D_a[:, :2000], y_axis='hz', x_axis='time', 
                                       sr=sr_a, ax=axes[0, 0])
        axes[0, 0].set_title('Track A Spectrogram')
        axes[0, 0].set_ylim(0, 8000)
        
        # Track B spectrogram
        img_b = librosa.display.specshow(D_b[:, :2000], y_axis='hz', x_axis='time', 
                                       sr=sr_b, ax=axes[0, 1])
        axes[0, 1].set_title('Track B Spectrogram')
        axes[0, 1].set_ylim(0, 8000)
        
        # Mix spectrogram (first part)
        img_mix = librosa.display.specshow(D_mix[:, :4000], y_axis='hz', x_axis='time', 
                                         sr=sr_mix, ax=axes[1, 0])
        axes[1, 0].set_title('Mix Spectrogram (Transition)')
        axes[1, 0].set_ylim(0, 8000)
        
        # Harmonic content comparison
        harm_a = librosa.effects.harmonic(y_a)
        harm_b = librosa.effects.harmonic(y_b)
        
        chroma_a = librosa.feature.chroma_stft(y=harm_a, sr=sr_a)
        chroma_b = librosa.feature.chroma_stft(y=harm_b, sr=sr_b)
        
        # Average chroma
        chroma_a_mean = np.mean(chroma_a, axis=1)
        chroma_b_mean = np.mean(chroma_b, axis=1)
        
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        x = np.arange(len(notes))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, chroma_a_mean, width, label='Track A', alpha=0.8)
        axes[1, 1].bar(x + width/2, chroma_b_mean, width, label='Track B', alpha=0.8)
        axes[1, 1].set_xlabel('Pitch Class')
        axes[1, 1].set_ylabel('Average Energy')
        axes[1, 1].set_title('Harmonic Content Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(notes)
        axes[1, 1].legend()
        
        plt.tight_layout()
        spectral_path = os.path.join(output_dir, 'spectral_analysis.png')
        plt.savefig(spectral_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualization_data['spectral_analysis'] = spectral_path
        
        # 5. Mix timeline visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create timeline
        total_duration = mix_script_data.get('total_duration', 180)
        
        # Track activity bars
        deck_a_active = []
        deck_b_active = []
        current_a_state = False
        current_b_state = False
        last_time = 0
        
        sorted_commands = sorted(mix_script_data.get('script', []), key=lambda x: x['time'])
        
        for cmd in sorted_commands:
            if cmd.get('params', {}).get('deck') == 'A':
                if cmd['command'] == 'play':
                    current_a_state = True
                elif cmd['command'] == 'stop':
                    if current_a_state:
                        deck_a_active.append((last_time, cmd['time']))
                    current_a_state = False
            elif cmd.get('params', {}).get('deck') == 'B':
                if cmd['command'] == 'play':
                    current_b_state = True
                elif cmd['command'] == 'stop':
                    if current_b_state:
                        deck_b_active.append((last_time, cmd['time']))
                    current_b_state = False
            last_time = cmd['time']
        
        # Add final segments if still playing
        if current_a_state:
            deck_a_active.append((last_time, total_duration))
        if current_b_state:
            deck_b_active.append((last_time, total_duration))
        
        # Plot track activity
        for start, end in deck_a_active:
            ax.barh(2, end - start, left=start, height=0.8, alpha=0.7, color='blue', label='Track A' if start == deck_a_active[0][0] else '')
        
        for start, end in deck_b_active:
            ax.barh(1, end - start, left=start, height=0.8, alpha=0.7, color='orange', label='Track B' if start == deck_b_active[0][0] else '')
        
        # Plot crossfader position
        crossfader_points = [(0, 0)]  # Start at A
        for cmd in sorted_commands:
            if cmd['command'] == 'set_crossfader':
                crossfader_points.append((cmd['time'], cmd['params']['position']))
        
        # Interpolate crossfader position
        if crossfader_points:
            times = [p[0] for p in crossfader_points]
            positions = [p[1] for p in crossfader_points]
            
            # Create smooth interpolation
            t_interp = np.linspace(0, total_duration, 1000)
            cf_interp = np.interp(t_interp, times, positions)
            
            ax.plot(t_interp, cf_interp * 0.8, 'k-', linewidth=2, label='Crossfader', alpha=0.8)
        
        # Mark key events
        event_types = {
            'set_hot_cue': 'H',
            'auto_loop': 'L',
            'toggle_pad_fx': 'FX',
            'beat_repeat': 'R',
            'set_beat_fx': 'BFX'
        }
        
        for cmd in sorted_commands:
            if cmd['command'] in event_types:
                y_pos = 2.5 if cmd.get('params', {}).get('deck') == 'A' else 1.5
                ax.plot(cmd['time'], y_pos, 'ro', markersize=8)
                ax.text(cmd['time'], y_pos + 0.1, event_types[cmd['command']], 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlim(0, total_duration)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Track / Crossfader')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Crossfader B', 'Track B', 'Track A'])
        ax.set_title('Mix Timeline and Events')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timeline_path = os.path.join(output_dir, 'mix_timeline.png')
        plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualization_data['mix_timeline'] = timeline_path
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
    
    return visualization_data