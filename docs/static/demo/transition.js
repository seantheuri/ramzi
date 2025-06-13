// API Configuration
const API_BASE = window.location.hostname === 'localhost' ? 'http://127.0.0.1:5000/api' : '/api';

// Global State
let transitionScript = null;
let renderedAudioUrl = null;
let audioContext = null;
let audioBuffer = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIConnection();
});

// Setup Event Listeners
function setupEventListeners() {
    // File inputs
    document.getElementById('transitionTrackA').addEventListener('change', (e) => {
        handleFileSelect(e, 'A');
    });
    
    document.getElementById('transitionTrackB').addEventListener('change', (e) => {
        handleFileSelect(e, 'B');
    });
    
    // Audio element for visualization sync
    const audio = document.getElementById('transitionAudio');
    audio.addEventListener('timeupdate', updateVisualization);
    audio.addEventListener('play', () => startVisualization());
    audio.addEventListener('pause', () => stopVisualization());
    
    // Drag and drop
    setupDragAndDrop();
}

// File Handling
function handleFileSelect(event, deck) {
    const file = event.target.files[0];
    if (!file) return;
    
    const box = event.target.closest('.upload-box');
    const label = box ? box.querySelector('.upload-area') : null;
    if (label) {
        label.classList.add('has-file');
        const uploadText = label.querySelector('.upload-text');
        if (uploadText) {
            uploadText.textContent = `✅ ${file.name}`;
        }
    }
    
    // Show file info
    const infoElement = document.getElementById(`track${deck}FileInfo`);
    infoElement.querySelector('.filename').textContent = file.name;
    infoElement.querySelector('.filesize').textContent = formatFileSize(file.size);
    infoElement.classList.add('show');
    
    checkTransitionReady();
}

function setupDragAndDrop() {
    document.querySelectorAll('.upload-area').forEach(area => {
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.style.borderColor = 'var(--highlight)';
        });
        
        area.addEventListener('dragleave', () => {
            area.style.borderColor = '';
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.style.borderColor = '';
            const input = area.previousElementSibling;
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                input.dispatchEvent(new Event('change'));
            }
        });
    });
}

function checkTransitionReady() {
    const ready = document.getElementById('transitionTrackA').files.length > 0 &&
                  document.getElementById('transitionTrackB').files.length > 0;
    
    document.getElementById('createTransitionBtn').disabled = !ready;
}

// Preset Prompts
function setPreset(preset) {
    const prompts = {
        smooth: 'Create a smooth 32-bar blend with gradual EQ swap and filter sweeps. Focus on harmonic mixing.',
        cut: 'Quick hip-hop style cut with transform effects and scratching. Make it energetic and punchy.',
        creative: 'Showcase advanced techniques with loops, hot cues, and creative effects. Surprise me with something unique.',
        harmonic: 'Focus on harmonic mixing with smooth key-matched transitions. Use pitch adjustment if needed.',
        buildup: 'Create tension with filters and effects, then drop into the new track with maximum impact.',
        mashup: 'Layer elements creatively, using loops and hot cues to create a unique blend. Keep both tracks recognizable.'
    };
    
    document.getElementById('transitionPrompt').value = prompts[preset] || '';
}

// Create Transition
async function createTransition() {
    const fileA = document.getElementById('transitionTrackA').files[0];
    const fileB = document.getElementById('transitionTrackB').files[0];
    const prompt = document.getElementById('transitionPrompt').value || 'Create a professional mix';
    
    // Show progress
    document.getElementById('progressSection').classList.add('active');
    document.getElementById('createTransitionBtn').disabled = true;
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('trackA', fileA);
        formData.append('trackB', fileB);
        formData.append('userPrompt', prompt);
        formData.append('mixStyle', 'blend');
        
        updateProgress(10, 'Uploading tracks...');
        
        // Call simplified API
        const response = await fetch(`${API_BASE}/create-mix-simple`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Mix creation failed');
        }
        
        updateProgress(40, 'Analyzing audio features...');
        
        const result = await response.json();
        transitionScript = result;
        
        updateProgress(70, 'Generating AI mix script...');
        
        // Render the mix
        const renderResponse = await fetch(`${API_BASE}/render-mix`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(result)
        });
        
        if (!renderResponse.ok) {
            throw new Error('Audio rendering failed');
        }
        
        updateProgress(90, 'Finalizing audio...');
        
        const renderResult = await renderResponse.json();
        renderedAudioUrl = renderResult.download_url;
        
        updateProgress(100, 'Complete!');
        
        // Display results
        setTimeout(() => {
            document.getElementById('progressSection').classList.remove('active');
            displayTransitionResults();
        }, 500);
        
    } catch (error) {
        document.getElementById('progressSection').classList.remove('active');
        document.getElementById('createTransitionBtn').disabled = false;
        showMessage(`Error: ${error.message}`, 'error');
    }
}

function updateProgress(percent, status) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('statusText').textContent = status;
}

// Display Results
async function displayTransitionResults() {
    document.getElementById('transitionResults').style.display = 'block';
    
    // Display description
    const description = transitionScript.description || 'Your AI-generated mix is ready!';
    document.getElementById('mixDescription').textContent = description;
    
    // Load rendered audio
    const audio = document.getElementById('transitionAudio');
    audio.src = renderedAudioUrl.startsWith('http') ? renderedAudioUrl : `${API_BASE.replace('/api', '')}${renderedAudioUrl}`;
    
    // Display track analysis
    if (transitionScript.trackA) {
        displayTrackAnalysis('A', transitionScript.trackA);
    }
    if (transitionScript.trackB) {
        displayTrackAnalysis('B', transitionScript.trackB);
    }
    
    // Display techniques
    displayTechniques(transitionScript.technique_highlights || []);
    
    // Draw waveform
    await drawTransitionWaveform();
    
    // Add script markers
    addScriptMarkers();
    
    // Scroll to results
    document.getElementById('transitionResults').scrollIntoView({ behavior: 'smooth' });
}

function displayTrackAnalysis(deck, data) {
    const container = document.getElementById(`track${deck}Analysis`);
    container.innerHTML = `
        <div class="info-card">
            <h4>BPM</h4>
            <p>${data.bpm || '-'}</p>
        </div>
        <div class="info-card">
            <h4>Key</h4>
            <p>${data.key_standard || '-'}</p>
        </div>
        <div class="info-card">
            <h4>Camelot</h4>
            <p>${data.key_camelot || '-'}</p>
        </div>
        <div class="info-card">
            <h4>Energy</h4>
            <p>${data.energy_normalized ? (parseFloat(data.energy_normalized) * 100).toFixed(0) + '%' : '-'}</p>
        </div>
    `;
}

function displayTechniques(techniques) {
    const container = document.getElementById('techniquesList');
    container.innerHTML = '';
    
    techniques.forEach(technique => {
        const tag = document.createElement('div');
        tag.style.cssText = `
            background: var(--accent-blue);
            padding: 10px 20px;
            border-radius: 20px;
            border: 1px solid var(--border-color);
        `;
        tag.textContent = technique;
        container.appendChild(tag);
    });
}

async function drawTransitionWaveform() {
    const canvas = document.getElementById('transitionWaveform');
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = 200;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Try to load the audio for proper waveform
    try {
        const audio = document.getElementById('transitionAudio');
        if (audio.src) {
            const response = await fetch(audio.src);
            const arrayBuffer = await response.arrayBuffer();
            
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Draw actual waveform
            const data = audioBuffer.getChannelData(0);
            const step = Math.ceil(data.length / width);
            const amp = height / 2;
            
            ctx.beginPath();
            ctx.moveTo(0, amp);
            ctx.strokeStyle = '#4a6a8e';
            ctx.lineWidth = 1;
            
            for (let i = 0; i < width; i++) {
                let min = 1.0;
                let max = -1.0;
                
                for (let j = 0; j < step; j++) {
                    const datum = data[(i * step) + j];
                    if (datum < min) min = datum;
                    if (datum > max) max = datum;
                }
                
                ctx.lineTo(i, (1 + min) * amp);
                ctx.lineTo(i, (1 + max) * amp);
            }
            
            ctx.stroke();
        }
    } catch (error) {
        // Fallback to placeholder waveform
        ctx.strokeStyle = '#4a6a8e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        for (let i = 0; i < width; i++) {
            const y = height / 2 + Math.sin(i * 0.02) * 30 * Math.random();
            if (i === 0) ctx.moveTo(i, y);
            else ctx.lineTo(i, y);
        }
        
        ctx.stroke();
    }
}

function addScriptMarkers() {
    if (!transitionScript || !transitionScript.script) return;
    
    const container = document.getElementById('markersContainer');
    container.innerHTML = '';
    
    const totalDuration = transitionScript.total_duration || 300;
    const width = document.getElementById('transitionWaveform').width;
    
    // Group commands by time for better visualization
    const commandsByTime = {};
    transitionScript.script.forEach(command => {
        const time = command.time.toFixed(1);
        if (!commandsByTime[time]) {
            commandsByTime[time] = [];
        }
        commandsByTime[time].push(command);
    });
    
    // Create markers
    Object.entries(commandsByTime).forEach(([time, commands]) => {
        const marker = document.createElement('div');
        marker.className = 'marker';
        marker.style.left = `${(parseFloat(time) / totalDuration) * width}px`;
        
        // Color code by command type
        const primaryCommand = commands[0].command;
        if (primaryCommand.includes('crossfader')) {
            marker.style.background = '#ff6b6b';
        } else if (primaryCommand.includes('eq') || primaryCommand.includes('filter')) {
            marker.style.background = '#4ecdc4';
        } else if (primaryCommand.includes('play') || primaryCommand.includes('stop')) {
            marker.style.background = '#ffd700';
        } else {
            marker.style.background = '#9b59b6';
        }
        
        const tooltip = document.createElement('div');
        tooltip.className = 'marker-tooltip';
        
        let tooltipHTML = `<strong>${formatTime(parseFloat(time))}</strong><br>`;
        commands.forEach(cmd => {
            tooltipHTML += `${cmd.command}`;
            if (cmd.params && cmd.params.deck) {
                tooltipHTML += ` (Deck ${cmd.params.deck})`;
            }
            tooltipHTML += '<br>';
        });
        
        tooltip.innerHTML = tooltipHTML;
        marker.appendChild(tooltip);
        
        // Click to seek
        marker.addEventListener('click', () => {
            const audio = document.getElementById('transitionAudio');
            audio.currentTime = parseFloat(time);
        });
        
        container.appendChild(marker);
    });
}

// Visualization
let visualizationActive = false;

function startVisualization() {
    visualizationActive = true;
    updateVisualization();
}

function stopVisualization() {
    visualizationActive = false;
}

function updateVisualization() {
    if (!visualizationActive) return;
    
    const audio = document.getElementById('transitionAudio');
    const playhead = document.getElementById('transitionPlayhead');
    const canvas = document.getElementById('transitionWaveform');
    
    if (audio.duration) {
        const progress = audio.currentTime / audio.duration;
        playhead.style.left = `${progress * canvas.width}px`;
    }
    
    if (!audio.paused) {
        requestAnimationFrame(updateVisualization);
    }
}

// Download Functions
function downloadMix() {
    if (renderedAudioUrl) {
        const a = document.createElement('a');
        a.href = renderedAudioUrl.startsWith('http') ? renderedAudioUrl : `${API_BASE.replace('/api', '')}${renderedAudioUrl}`;
        a.download = 'ramzi-mix.wav';
        a.click();
    }
}

function downloadScript() {
    if (transitionScript) {
        const blob = new Blob([JSON.stringify(transitionScript, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'mix-script.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}

function resetTransition() {
    // Reset form
    document.getElementById('transitionResults').style.display = 'none';
    document.getElementById('transitionTrackA').value = '';
    document.getElementById('transitionTrackB').value = '';
    document.getElementById('transitionPrompt').value = '';
    
    // Reset UI
    document.querySelectorAll('.upload-area').forEach(area => {
        area.classList.remove('has-file');
        const uploadText = area.querySelector('.upload-text');
        if (uploadText.textContent.includes('✅')) {
            uploadText.textContent = uploadText.textContent.replace('✅', '📁 Drop');
        }
    });
    
    document.querySelectorAll('.file-info').forEach(info => {
        info.classList.remove('show');
    });
    
    document.getElementById('createTransitionBtn').disabled = false;
    checkTransitionReady();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Utility Functions
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showMessage(text, type) {
    const message = document.getElementById(`${type}Message`);
    message.textContent = text;
    message.classList.add('show');
    
    setTimeout(() => {
        message.classList.remove('show');
    }, 5000);
}

async function checkAPIConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
            showMessage('API connection failed. Please check the server.', 'error');
        }
    } catch (error) {
        showMessage('Cannot connect to API. Make sure the server is running.', 'error');
    }
}