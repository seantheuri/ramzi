import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import threading
import logging

# Import our modules
from analysis import analyze_song
from llm import generate_mix_script, summarize_track_data, validate_mix_script
from controller import AudioRenderer

# Resolve project root (two levels up from this file: project/src/api.py -> project/)
BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(__name__)
CORS(app)  # Enable CORS for web app

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ANALYSIS_FOLDER'] = 'analysis'
app.config['MIX_SCRIPTS_FOLDER'] = 'mix_scripts'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Ensure directories exist
for folder in ['uploads', 'analysis', 'mix_scripts', 'rendered_mixes']:
    os.makedirs(BASE_DIR / folder, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for background tasks
background_tasks = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to run background tasks
def run_background_task(task_id: str, task_func, *args, **kwargs):
    """Run a task in background and store results"""
    try:
        background_tasks[task_id]['status'] = 'running'
        background_tasks[task_id]['progress'] = 0
        
        result = task_func(*args, **kwargs)
        
        background_tasks[task_id]['status'] = 'completed'
        background_tasks[task_id]['result'] = result
        background_tasks[task_id]['progress'] = 100
        
    except Exception as e:
        background_tasks[task_id]['status'] = 'failed'
        background_tasks[task_id]['error'] = str(e)
        logger.error(f"Background task {task_id} failed: {e}")

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'RAMZI DJ API is running',
        'version': '1.0.0'
    })

@app.route('/api/tracks', methods=['GET'])
def list_tracks():
    """List all available tracks and their analysis status"""
    tracks = []
    
    # Check uploads folder
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    analysis_folder = Path(app.config['ANALYSIS_FOLDER'])
    
    for audio_file in upload_folder.glob('*'):
        if audio_file.suffix.lower()[1:] in ALLOWED_EXTENSIONS:
            # Check if analysis exists
            analysis_file = analysis_folder / f"{audio_file.stem}_analysis.json"
            has_analysis = analysis_file.exists()
            
            # Load analysis data if available
            analysis_data = None
            if has_analysis:
                try:
                    with open(analysis_file, 'r') as f:
                        analysis_data = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading analysis for {audio_file}: {e}")
                    has_analysis = False
            
            tracks.append({
                'id': audio_file.stem,
                'filename': audio_file.name,
                'path': str(audio_file),
                'size': audio_file.stat().st_size,
                'has_analysis': has_analysis,
                'analysis_file': str(analysis_file) if has_analysis else None,
                'bpm': analysis_data.get('bpm') if analysis_data else None,
                'key': analysis_data.get('key_standard') if analysis_data else None,
                'energy': analysis_data.get('energy_normalized') if analysis_data else None
            })
    
    return jsonify({
        'tracks': tracks,
        'total': len(tracks)
    })

@app.route('/api/tracks/upload', methods=['POST'])
def upload_track():
    """Upload an audio file for analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = BASE_DIR / app.config['UPLOAD_FOLDER'] / filename
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'path': str(file_path),
            'id': Path(filename).stem
        })
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large'}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/api/tracks/<track_id>/analyze', methods=['POST'])
def analyze_track(track_id):
    """Start analysis of a track (background task)"""
    # Find the track file
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    track_file = None
    
    for audio_file in upload_folder.glob(f"{track_id}.*"):
        if audio_file.suffix.lower()[1:] in ALLOWED_EXTENSIONS:
            track_file = audio_file
            break
    
    if not track_file:
        return jsonify({'error': 'Track not found'}), 404
    
    # Check if analysis already exists
    analysis_file = Path(app.config['ANALYSIS_FOLDER']) / f"{track_id}_analysis.json"
    if analysis_file.exists():
        return jsonify({
            'message': 'Analysis already exists',
            'analysis_file': str(analysis_file),
            'track_id': track_id
        })
    
    # Get options from request
    include_lyrics = request.json.get('include_lyrics', True) if request.json else True
    
    # Create background task
    task_id = str(uuid.uuid4())
    background_tasks[task_id] = {
        'status': 'queued',
        'track_id': track_id,
        'task_type': 'analysis',
        'created_at': time.time(),
        'progress': 0
    }
    
    # Define analysis task
    def analysis_task():
        try:
            # Update progress
            background_tasks[task_id]['progress'] = 10
            background_tasks[task_id]['status'] = 'running'
            
            result = analyze_song(str(track_file), include_lyrics=include_lyrics)
            
            if result:
                # Save analysis
                background_tasks[task_id]['progress'] = 90
                output_path = Path(app.config['ANALYSIS_FOLDER']) / f"{track_id}_analysis.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                background_tasks[task_id]['progress'] = 100
                background_tasks[task_id]['status'] = 'completed'
                background_tasks[task_id]['result'] = {
                    'analysis_file': str(output_path),
                    'analysis_data': result
                }
            else:
                raise Exception("Analysis returned no results")
                
        except Exception as e:
            background_tasks[task_id]['status'] = 'failed'
            background_tasks[task_id]['error'] = str(e)
            logger.error(f"Analysis task {task_id} failed: {e}")
    
    # Start background task
    thread = threading.Thread(target=analysis_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Analysis started',
        'task_id': task_id,
        'track_id': track_id
    })

@app.route('/api/tracks/<track_id>/analysis', methods=['GET'])
def get_track_analysis(track_id):
    """Get analysis data for a track"""
    analysis_file = Path(app.config['ANALYSIS_FOLDER']) / f"{track_id}_analysis.json"
    
    if not analysis_file.exists():
        return jsonify({'error': 'Analysis not found'}), 404
    
    try:
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        # Add summary data
        summary = summarize_track_data(analysis_data)
        
        return jsonify({
            'track_id': track_id,
            'analysis_file': str(analysis_file),
            'analysis_data': analysis_data,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error loading analysis for {track_id}: {e}")
        return jsonify({'error': 'Failed to load analysis'}), 500

@app.route('/api/mix/generate', methods=['POST'])
def generate_mix():
    """Generate a mix script using LLM"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    track_a_id = data.get('track_a_id')
    track_b_id = data.get('track_b_id')
    user_prompt = data.get('user_prompt', 'Create a professional mix')
    mix_style = data.get('mix_style', 'blend')
    model = data.get('model', 'gpt-4o')
    
    if not track_a_id or not track_b_id:
        return jsonify({'error': 'Both track_a_id and track_b_id are required'}), 400
    
    # Check if analysis files exist
    analysis_folder = Path(app.config['ANALYSIS_FOLDER'])
    analysis_file_a = analysis_folder / f"{track_a_id}_analysis.json"
    analysis_file_b = analysis_folder / f"{track_b_id}_analysis.json"
    
    if not analysis_file_a.exists():
        return jsonify({'error': f'Analysis not found for track A: {track_a_id}'}), 404
    
    if not analysis_file_b.exists():
        return jsonify({'error': f'Analysis not found for track B: {track_b_id}'}), 404
    
    # Create background task
    task_id = str(uuid.uuid4())
    background_tasks[task_id] = {
        'status': 'queued',
        'task_type': 'mix_generation',
        'track_a_id': track_a_id,
        'track_b_id': track_b_id,
        'created_at': time.time(),
        'progress': 0
    }
    
    # Define mix generation task
    def mix_generation_task():
        try:
            background_tasks[task_id]['status'] = 'running'
            background_tasks[task_id]['progress'] = 10
            
            # Generate unique output filename
            output_filename = f"mix_{track_a_id}_to_{track_b_id}_{int(time.time())}.json"
            output_path = BASE_DIR / app.config['MIX_SCRIPTS_FOLDER'] / output_filename
            
            background_tasks[task_id]['progress'] = 30
            
            # Generate mix script
            generate_mix_script(
                str(analysis_file_a),
                str(analysis_file_b),
                user_prompt,
                str(output_path),
                mix_style,
                model
            )
            
            background_tasks[task_id]['progress'] = 90
            
            # Load and validate the generated script
            with open(output_path, 'r') as f:
                mix_script = json.load(f)
            
            is_valid = validate_mix_script(mix_script)
            
            background_tasks[task_id]['progress'] = 100
            background_tasks[task_id]['status'] = 'completed'
            background_tasks[task_id]['result'] = {
                'mix_script_file': str(output_path),
                'mix_script': mix_script,
                'is_valid': is_valid
            }
            
        except Exception as e:
            background_tasks[task_id]['status'] = 'failed'
            background_tasks[task_id]['error'] = str(e)
            logger.error(f"Mix generation task {task_id} failed: {e}")
    
    # Start background task
    thread = threading.Thread(target=mix_generation_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Mix generation started',
        'task_id': task_id,
        'track_a_id': track_a_id,
        'track_b_id': track_b_id
    })

@app.route('/api/mix/scripts', methods=['GET'])
def list_mix_scripts():
    """List all available mix scripts"""
    scripts = []
    mix_scripts_folder = Path(app.config['MIX_SCRIPTS_FOLDER'])
    
    for script_file in mix_scripts_folder.glob('*.json'):
        try:
            with open(script_file, 'r') as f:
                script_data = json.load(f)
            
            # Extract metadata from filename and content
            file_parts = script_file.stem.split('_')
            
            scripts.append({
                'id': script_file.stem,
                'filename': script_file.name,
                'path': str(script_file),
                'created_at': script_file.stat().st_mtime,
                'description': script_data.get('description', ''),
                'total_duration': script_data.get('total_duration', 0),
                'technique_highlights': script_data.get('technique_highlights', []),
                'command_count': len(script_data.get('script', []))
            })
            
        except Exception as e:
            logger.error(f"Error reading mix script {script_file}: {e}")
    
    # Sort by creation time (newest first)
    scripts.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({
        'scripts': scripts,
        'total': len(scripts)
    })

@app.route('/api/mix/scripts/<script_id>', methods=['GET'])
def get_mix_script(script_id):
    """Get a specific mix script"""
    script_file = Path(app.config['MIX_SCRIPTS_FOLDER']) / f"{script_id}.json"
    
    if not script_file.exists():
        return jsonify({'error': 'Mix script not found'}), 404
    
    try:
        with open(script_file, 'r') as f:
            script_data = json.load(f)
        
        return jsonify({
            'script_id': script_id,
            'script_file': str(script_file),
            'script_data': script_data
        })
        
    except Exception as e:
        logger.error(f"Error loading mix script {script_id}: {e}")
        return jsonify({'error': 'Failed to load mix script'}), 500

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a background task"""
    if task_id not in background_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = background_tasks[task_id]
    
    # Clean up old completed tasks (older than 1 hour)
    if task['status'] in ['completed', 'failed'] and time.time() - task['created_at'] > 3600:
        del background_tasks[task_id]
        return jsonify({'error': 'Task expired'}), 410
    
    return jsonify(task)

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List all active background tasks"""
    # Clean up old tasks
    current_time = time.time()
    expired_tasks = [
        task_id for task_id, task in background_tasks.items()
        if task['status'] in ['completed', 'failed'] and current_time - task['created_at'] > 3600
    ]
    
    for task_id in expired_tasks:
        del background_tasks[task_id]
    
    return jsonify({
        'tasks': background_tasks,
        'total': len(background_tasks)
    })

@app.route('/api/files/<path:filename>', methods=['GET'])
def serve_file(filename):
    """Serve uploaded files, analysis files, mix scripts, or rendered audio"""
    # Determine which folder based on file extension
    if filename.endswith('_analysis.json'):
        folder = app.config['ANALYSIS_FOLDER']
    elif filename.endswith('.json'):
        folder = app.config['MIX_SCRIPTS_FOLDER']
    elif filename.endswith('.wav') or filename.endswith('.mp3'):
        folder = 'rendered_mixes'
    else:
        folder = app.config['UPLOAD_FOLDER']
    
    # Build absolute path relative to the project root to avoid CWD issues
    file_path = (BASE_DIR / folder / filename).resolve()
    
    if not file_path.exists():
        return jsonify({'error': f'File not found: {file_path}'}), 404
    
    return send_file(file_path)

@app.route('/api/create-mix-simple', methods=['POST'])
def create_mix_simple():
    """Simplified endpoint that handles the entire mix creation workflow in one call"""
    if 'trackA' not in request.files or 'trackB' not in request.files:
        return jsonify({'error': 'Both trackA and trackB files are required'}), 400
    
    file_a = request.files['trackA']
    file_b = request.files['trackB']
    
    if file_a.filename == '' or file_b.filename == '':
        return jsonify({'error': 'Both files must be selected'}), 400
    
    if not allowed_file(file_a.filename) or not allowed_file(file_b.filename):
        return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Get optional parameters
    user_prompt = request.form.get('userPrompt', 'Create a professional mix')
    mix_style = request.form.get('mixStyle', 'blend')
    model = request.form.get('model', 'gpt-4o')
    
    try:
        import tempfile
        import shutil
        
        # Create temporary directory for this processing session
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save both files temporarily
            logger.info("💾 Saving uploaded files...")
            track_a_path = temp_path / f"track_a_{secure_filename(file_a.filename)}"
            track_b_path = temp_path / f"track_b_{secure_filename(file_b.filename)}"
            
            file_a.save(str(track_a_path))
            file_b.save(str(track_b_path))
            
            # Analyze both tracks
            logger.info("🎵 Starting analysis of Track A...")
            analysis_a = analyze_song(str(track_a_path), include_lyrics=False)
            if not analysis_a:
                raise Exception("Failed to analyze Track A")
            
            logger.info("🎵 Starting analysis of Track B...")
            analysis_b = analyze_song(str(track_b_path), include_lyrics=False)
            if not analysis_b:
                raise Exception("Failed to analyze Track B")
            
            # Save analysis files temporarily for LLM processing
            analysis_a_path = temp_path / "track_a_analysis.json"
            analysis_b_path = temp_path / "track_b_analysis.json"
            
            with open(analysis_a_path, 'w') as f:
                json.dump(analysis_a, f, indent=2)
            with open(analysis_b_path, 'w') as f:
                json.dump(analysis_b, f, indent=2)
            
            # Generate mix script
            logger.info("🤖 Generating AI mix script...")
            mix_script_path = temp_path / "generated_mix.json"
            
            generate_mix_script(
                str(analysis_a_path),
                str(analysis_b_path),
                user_prompt,
                str(mix_script_path),
                mix_style,
                model
            )
            
            # Load the generated mix script
            with open(mix_script_path, 'r') as f:
                mix_script_data = json.load(f)
            
            # Validate the script
            is_valid = validate_mix_script(mix_script_data)
            if not is_valid:
                logger.warning("Generated mix script failed validation, but continuing...")
            
            # Create track summaries for response
            summary_a = summarize_track_data(analysis_a)
            summary_b = summarize_track_data(analysis_b)
            
            # Prepare complete response
            response_data = {
                'success': True,
                'description': mix_script_data.get('description', 'Your AI-generated mix is ready!'),
                'technique_highlights': mix_script_data.get('technique_highlights', []),
                'total_duration': mix_script_data.get('total_duration', 0),
                'script': mix_script_data.get('script', []),
                'trackA': summary_a,
                'trackB': summary_b,
                'mix_style': mix_style,
                'user_prompt': user_prompt,
                'is_valid': is_valid,
                'command_count': len(mix_script_data.get('script', [])),
                'processing_info': {
                    'tracks_analyzed': 2,
                    'analysis_duration': 'varies',
                    'ai_model_used': model
                }
            }
            
            logger.info(f"✅ Mix creation successful! Generated script with {len(mix_script_data.get('script', []))} commands")
            return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"❌ Mix creation failed: {e}")
        return jsonify({
            'error': f'Mix creation failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/render-mix', methods=['POST'])
def render_mix():
    """Render audio from a mix script JSON"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No mix script data provided'}), 400
        
        # Validate mix script structure
        if 'script' not in data or not isinstance(data['script'], list):
            return jsonify({'error': 'Invalid mix script format - missing script array'}), 400
        
        if 'total_duration' not in data:
            return jsonify({'error': 'Invalid mix script format - missing total_duration'}), 400
        
        logger.info(f"🎛️ Starting audio rendering for mix script with {len(data['script'])} commands")
        
        # Create unique output filename
        import time
        timestamp = int(time.time())
        output_filename = f"rendered_mix_{timestamp}.wav"
        output_path = (BASE_DIR / 'rendered_mixes' / output_filename).resolve()
        
        # Ensure output directory exists
        output_path.parent.mkdir(exist_ok=True)
        
        # Initialize audio renderer
        renderer = AudioRenderer(sample_rate=44100)
        
        # Render the mix
        logger.info("🎵 Rendering audio - this may take several minutes...")
        renderer.render(data, str(output_path))
        
        # Check if file was created successfully
        if not output_path.exists():
            raise Exception("Audio rendering failed - no output file created")
        
        file_size = output_path.stat().st_size
        duration = data['total_duration']
        
        logger.info(f"✅ Audio rendering completed! File: {output_path} ({file_size} bytes)")
        
        return jsonify({
            'success': True,
            'message': 'Mix rendered successfully',
            'output_file': str(output_path),
            'filename': output_filename,
            'file_size': file_size,
            'duration': duration,
            'download_url': f'/api/files/{output_filename}',
            'commands_processed': len(data['script'])
        })
        
    except Exception as e:
        logger.error(f"❌ Mix rendering failed: {e}")
        return jsonify({
            'error': f'Mix rendering failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get API configuration and capabilities"""
    return jsonify({
        'max_file_size': app.config['MAX_CONTENT_LENGTH'],
        'allowed_extensions': list(ALLOWED_EXTENSIONS),
        'mix_styles': ['blend', 'cut', 'creative', 'harmonic', 'buildup', 'mashup'],
        'openai_models': ['gpt-4o', 'gpt-4-turbo-preview', 'gpt-3.5-turbo'],
        'features': {
            'track_upload': True,
            'audio_analysis': True,
            'lyrics_transcription': True,
            'mix_generation': True,
            'background_tasks': True,
            'simplified_workflow': True,
            'audio_rendering': True
        }
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
