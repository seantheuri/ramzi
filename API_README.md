# RAMZI DJ API Documentation

A comprehensive REST API for the RAMZI Virtual DJ system, providing audio analysis, AI-powered mix generation, and track management capabilities.

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export SECRET_KEY="your-secret-key"  # Optional, for production
```

3. Start the API server:
```bash
python src/api.py
```

The API will be available at `http://localhost:5000`

### Basic Health Check
```bash
curl http://localhost:5000/api/health
```

## API Endpoints

### 🎵 Track Management

#### Upload Track
```http
POST /api/tracks/upload
Content-Type: multipart/form-data

file: audio_file.mp3
```

**Response:**
```json
{
  "message": "File uploaded successfully",
  "filename": "track.mp3",
  "path": "uploads/track.mp3",
  "id": "track"
}
```

#### List All Tracks
```http
GET /api/tracks
```

**Response:**
```json
{
  "tracks": [
    {
      "id": "track1",
      "filename": "track1.mp3",
      "path": "uploads/track1.mp3",
      "size": 5242880,
      "has_analysis": true,
      "analysis_file": "analysis/track1_analysis.json",
      "bpm": "128.5",
      "key": "A Minor",
      "energy": "0.750"
    }
  ],
  "total": 1
}
```

### 🔍 Track Analysis

#### Start Track Analysis
```http
POST /api/tracks/{track_id}/analyze
Content-Type: application/json

{
  "include_lyrics": true
}
```

**Response:**
```json
{
  "message": "Analysis started",
  "task_id": "uuid-string",
  "track_id": "track1"
}
```

#### Get Track Analysis
```http
GET /api/tracks/{track_id}/analysis
```

**Response:**
```json
{
  "track_id": "track1",
  "analysis_file": "analysis/track1_analysis.json",
  "analysis_data": {
    "file_path": "uploads/track1.mp3",
    "bpm": "128.5",
    "key_standard": "A Minor",
    "key_camelot": "8A",
    "energy_normalized": "0.750",
    "beat_grid_seconds": [0.5, 1.0, 1.5, 2.0],
    "structural_analysis": [
      {
        "label": "intro",
        "start": 0.0,
        "end": 32.0
      }
    ],
    "lyrics_timed": [
      {
        "word": "hello",
        "start": 10.5,
        "end": 11.0
      }
    ]
  },
  "summary": {
    "track_name": "track1.mp3",
    "bpm": "128.5",
    "key_standard": "A Minor",
    "key_camelot": "8A",
    "energy_normalized": "0.750",
    "beat_info": {
      "first_beat": 0.5,
      "beat_count": 100,
      "average_beat_interval": 0.47
    },
    "lyrics_sections": [],
    "total_duration": 180.0
  }
}
```

### 🎚️ Mix Generation

#### Generate Mix Script
```http
POST /api/mix/generate
Content-Type: application/json

{
  "track_a_id": "track1",
  "track_b_id": "track2",
  "user_prompt": "Create an energetic festival-style mix",
  "mix_style": "creative",
  "model": "gpt-4o"
}
```

**Response:**
```json
{
  "message": "Mix generation started",
  "task_id": "uuid-string",
  "track_a_id": "track1",
  "track_b_id": "track2"
}
```

#### List Mix Scripts
```http
GET /api/mix/scripts
```

**Response:**
```json
{
  "scripts": [
    {
      "id": "mix_track1_to_track2_1234567890",
      "filename": "mix_track1_to_track2_1234567890.json",
      "path": "mix_scripts/mix_track1_to_track2_1234567890.json",
      "created_at": 1234567890,
      "description": "Creative festival-style transition with loops and effects",
      "total_duration": 240.0,
      "technique_highlights": ["Hot cues", "Looping", "Beat FX"],
      "command_count": 25
    }
  ],
  "total": 1
}
```

#### Get Mix Script
```http
GET /api/mix/scripts/{script_id}
```

**Response:**
```json
{
  "script_id": "mix_track1_to_track2_1234567890",
  "script_file": "mix_scripts/mix_track1_to_track2_1234567890.json",
  "script_data": {
    "description": "Creative festival-style transition",
    "technique_highlights": ["Hot cues", "Looping", "Beat FX"],
    "total_duration": 240.0,
    "script": [
      {
        "time": 0.0,
        "command": "load_track",
        "params": {
          "deck": "A",
          "file_path": "analysis/track1_analysis.json",
          "target_bpm": 128
        }
      }
    ]
  }
}
```

### 📊 Background Tasks

#### Get Task Status
```http
GET /api/tasks/{task_id}
```

**Response:**
```json
{
  "status": "completed",
  "track_id": "track1",
  "task_type": "analysis",
  "created_at": 1234567890,
  "progress": 100,
  "result": {
    "analysis_file": "analysis/track1_analysis.json",
    "analysis_data": {...}
  }
}
```

#### List All Tasks
```http
GET /api/tasks
```

**Response:**
```json
{
  "tasks": {
    "uuid-1": {
      "status": "running",
      "progress": 50,
      "task_type": "analysis"
    },
    "uuid-2": {
      "status": "completed",
      "progress": 100,
      "task_type": "mix_generation"
    }
  },
  "total": 2
}
```

### 📁 File Serving

#### Download Files
```http
GET /api/files/{filename}
```

Examples:
- `/api/files/track1.mp3` - Download uploaded audio file
- `/api/files/track1_analysis.json` - Download analysis file
- `/api/files/mix_script.json` - Download mix script

### ⚙️ Configuration

#### Get API Configuration
```http
GET /api/config
```

**Response:**
```json
{
  "max_file_size": 104857600,
  "allowed_extensions": ["mp3", "wav", "flac", "m4a", "aac"],
  "mix_styles": ["blend", "cut", "creative", "harmonic", "buildup", "mashup"],
  "openai_models": ["gpt-4o", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
  "features": {
    "track_upload": true,
    "audio_analysis": true,
    "lyrics_transcription": true,
    "mix_generation": true,
    "background_tasks": true
  }
}
```

## Mix Styles

The API supports different mix styles:

- **blend**: Long, smooth blends where both tracks play together
- **cut**: Quick cuts and transforms for energetic mixes
- **creative**: Advanced techniques like beat juggling, loops, and effects
- **harmonic**: Key-compatible mixing with smooth transitions
- **buildup**: Tension and release using filters and effects
- **mashup**: Creative layering using loops and hot cues

## Background Tasks

Long-running operations (analysis and mix generation) run as background tasks:

1. **Start operation** → Get `task_id`
2. **Poll task status** → Check progress
3. **Retrieve results** → Get final output

### Task Status Values
- `queued`: Task is waiting to start
- `running`: Task is currently executing
- `completed`: Task finished successfully
- `failed`: Task encountered an error

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (resource doesn't exist)
- `413`: Payload Too Large (file too big)
- `500`: Internal Server Error

Error responses include details:
```json
{
  "error": "Description of what went wrong"
}
```

## File Limitations

- **Max file size**: 100MB
- **Supported formats**: MP3, WAV, FLAC, M4A, AAC
- **Analysis requirements**: OpenAI API key for lyrics transcription

## Web App Integration Examples

### JavaScript/TypeScript

#### Upload and Analyze Track
```javascript
// Upload file
const formData = new FormData();
formData.append('file', audioFile);

const uploadResponse = await fetch('/api/tracks/upload', {
  method: 'POST',
  body: formData
});

const { id } = await uploadResponse.json();

// Start analysis
const analysisResponse = await fetch(`/api/tracks/${id}/analyze`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ include_lyrics: true })
});

const { task_id } = await analysisResponse.json();

// Poll for completion
const pollTask = async () => {
  const taskResponse = await fetch(`/api/tasks/${task_id}`);
  const task = await taskResponse.json();
  
  if (task.status === 'completed') {
    console.log('Analysis complete:', task.result);
  } else if (task.status === 'running') {
    console.log('Progress:', task.progress + '%');
    setTimeout(pollTask, 1000); // Poll every second
  }
};

pollTask();
```

#### Generate Mix
```javascript
const generateMix = async (trackAId, trackBId) => {
  const response = await fetch('/api/mix/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      track_a_id: trackAId,
      track_b_id: trackBId,
      user_prompt: 'Create an energetic mix',
      mix_style: 'creative'
    })
  });
  
  const { task_id } = await response.json();
  return task_id;
};
```

### React Component Example
```jsx
import React, { useState, useEffect } from 'react';

const TrackAnalyzer = () => {
  const [tracks, setTracks] = useState([]);
  const [analysisTask, setAnalysisTask] = useState(null);

  // Load tracks
  useEffect(() => {
    fetch('/api/tracks')
      .then(res => res.json())
      .then(data => setTracks(data.tracks));
  }, []);

  // Upload handler
  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/tracks/upload', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    console.log('Uploaded:', result);
  };

  return (
    <div>
      <input 
        type="file" 
        accept=".mp3,.wav,.flac"
        onChange={(e) => handleUpload(e.target.files[0])}
      />
      
      <div>
        {tracks.map(track => (
          <div key={track.id}>
            <h3>{track.filename}</h3>
            <p>BPM: {track.bpm || 'Unknown'}</p>
            <p>Key: {track.key || 'Unknown'}</p>
            <p>Analysis: {track.has_analysis ? '✅' : '❌'}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
```

## Security Considerations

- Set `SECRET_KEY` environment variable in production
- Consider rate limiting for public deployments
- Validate file types and sizes on upload
- Sanitize file names (already handled by `secure_filename`)

## Development Tips

1. Use the `/api/health` endpoint to verify API connectivity
2. Check `/api/config` to understand API capabilities
3. Monitor background tasks using `/api/tasks`
4. Files are automatically organized into appropriate folders
5. Old completed tasks are automatically cleaned up after 1 hour 