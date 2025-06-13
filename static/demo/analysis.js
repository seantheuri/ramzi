// API Configuration
// Determine API base URL more robustly
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'https://ramzi-3c37.onrender.com:5000/api'
  : '/api';

// Global State
let analysisMode = 'single';
let audioContext = null;
let audioBuffers = {};
let audioSources = {};
let isPlaying = {};
let startTimes = {};
let pauseTimes = {};
let analysisData = {};
let trackIds = {}; // Store track IDs for API calls

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIConnection();
});

// Setup Event Listeners
function setupEventListeners() {
    // File inputs
    document.getElementById('singleTrackInput').addEventListener('change', (e) => {
        handleFileSelect(e, 'single');
    });
    
    document.getElementById('compareTrackA').addEventListener('change', (e) => {
        handleFileSelect(e, 'compareA');
    });
    
    document.getElementById('compareTrackB').addEventListener('change', (e) => {
        handleFileSelect(e, 'compareB');
    });
    
    // Drag and drop
    setupDragAndDrop();
}

// Mode Selection
function setAnalysisMode(mode) {
    analysisMode = mode;
    
    // Update UI
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    if (mode === 'single') {
        document.getElementById('singleTrackMode').style.display = 'grid';
        document.getElementById('compareTracksMode').style.display = 'none';
    } else {
        document.getElementById('singleTrackMode').style.display = 'none';
        document.getElementById('compareTracksMode').style.display = 'grid';
    }
    
    // Reset results
    document.getElementById('singleTrackResults').style.display = 'none';
    document.getElementById('compareResults').style.display = 'none';
    
    checkAnalysisReady();
}

// File Handling
function handleFileSelect(event, type) {
    const file = event.target.files[0];
    if (!file) return;

    const input = event.target;
    // Depending on markup order the label can be previous or next sibling.
    let label = input.previousElementSibling;
    if (!label || !label.classList.contains('upload-area')) {
        label = input.nextElementSibling;
    }

    if (!label || !label.classList.contains('upload-area')) {
        console.error('Upload label element not found for input:', input);
    } else {
        label.classList.add('has-file');
        const uploadText = label.querySelector('.upload-text');
        if (uploadText) {
            uploadText.textContent = `✅ ${file.name}`;
        }
    }

    // Determine the info panel that should display this file's metadata based on context.
    let infoElement;
    if (type === 'single') {
        infoElement = document.getElementById('singleFileInfo');
    } else if (type === 'compareA') {
        infoElement = document.getElementById('trackAInfo');
    } else {
        infoElement = document.getElementById('trackBInfo');
    }

    // If the info panel exists, populate it with file details.
    if (infoElement) {
        const filenameEl = infoElement.querySelector('.filename');
        const filesizeEl = infoElement.querySelector('.filesize');
        const filetypeEl = infoElement.querySelector('.filetype');

        if (filenameEl) filenameEl.textContent = file.name;
        if (filesizeEl) filesizeEl.textContent = formatFileSize(file.size);
        if (filetypeEl) filetypeEl.textContent = file.type || 'Unknown';

        infoElement.classList.add('show');
    }

    // Re-evaluate whether the user has provided all required files to enable analysis.
    checkAnalysisReady();
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

function checkAnalysisReady() {
    let ready = false;
    
    if (analysisMode === 'single') {
        ready = document.getElementById('singleTrackInput').files.length > 0;
    } else {
        ready = document.getElementById('compareTrackA').files.length > 0 &&
                document.getElementById('compareTrackB').files.length > 0;
    }
    
    document.getElementById('analyzeBtn').disabled = !ready;
}

// Analysis Functions
async function startAnalysis() {
    showMessage('Starting analysis...', 'success');
    document.getElementById('progressSection').classList.add('active');
    
    try {
        if (analysisMode === 'single') {
            const file = document.getElementById('singleTrackInput').files[0];
            updateProgress(10, 'Uploading track...');
            
            const result = await analyzeTrack(file, 'single');
            
            updateProgress(100, 'Analysis complete!');
            await loadAudioFile(file, 'single');
            
            setTimeout(() => {
                document.getElementById('progressSection').classList.remove('active');
                displaySingleTrackResults(result);
            }, 500);
        } else {
            const fileA = document.getElementById('compareTrackA').files[0];
            const fileB = document.getElementById('compareTrackB').files[0];
            
            updateProgress(10, 'Uploading tracks...');
            
            const [resultA, resultB] = await Promise.all([
                analyzeTrack(fileA, 'A'),
                analyzeTrack(fileB, 'B')
            ]);
            
            updateProgress(90, 'Loading audio and comparing...');
            
            await Promise.all([
                loadAudioFile(fileA, 'A'),
                loadAudioFile(fileB, 'B')
            ]);
            
            const compatibility = await getCompatibilityAnalysis(trackIds.A, trackIds.B);
            
            updateProgress(100, 'Analysis complete!');
            
            setTimeout(() => {
                document.getElementById('progressSection').classList.remove('active');
                displayCompareResults(resultA, resultB, compatibility);
            }, 500);
        }
    } catch (error) {
        document.getElementById('progressSection').classList.remove('active');
        showMessage(`Analysis failed: ${error.message}`, 'error');
    }
}

async function analyzeTrack(file, trackId) {
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Upload track
    updateProgress(20, `Uploading ${file.name}...`);
    const uploadResponse = await fetch(`${API_BASE}/tracks/upload`, {
        method: 'POST',
        body: formData
    });
    
    if (!uploadResponse.ok) {
        throw new Error('Upload failed');
    }
    
    const uploadData = await uploadResponse.json();
    const apiTrackId = uploadData.id;
    trackIds[trackId] = apiTrackId;
    
    // Start analysis
    updateProgress(40, 'Analyzing audio features...');
    const analysisResponse = await fetch(`${API_BASE}/tracks/${apiTrackId}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_lyrics: false })
    });
    
    if (!analysisResponse.ok) {
        throw new Error('Analysis failed');
    }
    
    const analysisTask = await analysisResponse.json();
    
    if (analysisTask.task_id) {
        // New analysis started – poll task endpoint
        updateProgress(60, 'Processing BPM and key detection...');
        await pollTask(analysisTask.task_id);
    } else if (analysisTask.analysis_file) {
        // Analysis already existed – skip polling
        console.info('Analysis already exists, skipping task polling.');
    } else {
        throw new Error('Server did not return a valid task ID or analysis file.');
    }
    
    // Get final analysis
    updateProgress(80, 'Finalizing results...');
    const resultsResponse = await fetch(`${API_BASE}/tracks/${apiTrackId}/analysis`);
    const results = await resultsResponse.json();
    
    return results;
}

async function getCompatibilityAnalysis(idA, idB) {
    try {
        const res = await fetch(`${API_BASE}/tracks/compare`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ track_a_id: idA, track_b_id: idB })
        });
        return res.ok ? await res.json() : null;
    } catch (e) {
        console.warn('Compatibility fetch error', e);
        return null;
    }
}

async function getAdvancedFeatures(trackId) {
    try { const r = await fetch(`${API_BASE}/tracks/${trackId}/features`); return r.ok ? await r.json() : null; }
    catch(e) { console.warn('Advanced feature error', e); return null; }
}

async function pollTask(taskId) {
    return new Promise((resolve, reject) => {
        const poll = async () => {
            try {
                const response = await fetch(`${API_BASE}/tasks/${taskId}`);
                const task = await response.json();
                
                if (task.status === 'completed') {
                    resolve(task.result);
                } else if (task.status === 'failed') {
                    reject(new Error(task.error || 'Task failed'));
                } else {
                    setTimeout(poll, 1000);
                }
            } catch (error) {
                reject(error);
            }
        };
        poll();
    });
}

async function loadAudioFile(file, trackId) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    const arrayBuffer = await file.arrayBuffer();
    audioBuffers[trackId] = await audioContext.decodeAudioData(arrayBuffer);
}

function updateProgress(percent, status) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('statusText').textContent = status;
}

// Display Results
async function displaySingleTrackResults(results) {
    analysisData.single = results.analysis_data;
    if (trackIds.single) {
        const adv = await getAdvancedFeatures(trackIds.single);
        if (adv) analysisData.single.advanced_features = adv.features;
    }
    const track = analysisData.single.track || analysisData.single;
    const info = document.getElementById('trackInfo');
    info.innerHTML = `
        <div class="info-card"><h4>BPM</h4><p>${track.tempo || analysisData.single.bpm}</p><small>Conf: ${(track.tempo_confidence*100||85).toFixed(0)}%</small></div>
        <div class="info-card"><h4>Key</h4><p>${analysisData.single.key_standard}</p><small>Conf: ${(track.key_confidence*100||75).toFixed(0)}%</small></div>
        <div class="info-card"><h4>Camelot</h4><p>${analysisData.single.key_camelot||'N/A'}</p></div>
        <div class="info-card"><h4>Energy</h4><p>${(parseFloat(analysisData.single.energy_normalized)*100).toFixed(0)}%</p></div>
        <div class="info-card"><h4>Time Sig</h4><p>${track.time_signature||4}/4</p><small>Conf: ${(track.time_signature_confidence*100||80).toFixed(0)}%</small></div>
        <div class="info-card"><h4>Loudness</h4><p>${(track.loudness||-12).toFixed(1)} dB</p></div>`;
    if (analysisData.single.advanced_features) displayAdvancedFeatures(analysisData.single.advanced_features);
    displaySegments(analysisData.single.structural_analysis);
    drawWaveform('analysisWaveform', audioBuffers.single, analysisData.single);
    document.getElementById('singleTrackResults').style.display = 'block';
}

async function displayCompareResults(rA, rB, compat) {
    analysisData.A = rA.analysis_data; analysisData.B = rB.analysis_data;
    document.getElementById('trackInfoA').innerHTML = createTrackInfoHTML(analysisData.A);
    document.getElementById('trackInfoB').innerHTML = createTrackInfoHTML(analysisData.B);
    displayEnhancedCompatibility(compat || {});
    drawWaveform('waveformA', audioBuffers.A, analysisData.A);
    drawWaveform('waveformB', audioBuffers.B, analysisData.B);
    document.getElementById('compareResults').style.display = 'block';
}

function createTrackInfoHTML(d) {
    const t = d.track || d;
    return `
        <div class="info-card"><h4>BPM</h4><p>${t.tempo||d.bpm}</p></div>
        <div class="info-card"><h4>Key</h4><p>${d.key_standard}</p></div>
        <div class="info-card"><h4>Camelot</h4><p>${d.key_camelot||'N/A'}</p></div>
        <div class="info-card"><h4>Energy</h4><p>${(parseFloat(d.energy_normalized)*100).toFixed(0)}%</p></div>
        <div class="info-card"><h4>Time Sig</h4><p>${t.time_signature||4}/4</p></div>
        <div class="info-card"><h4>Loudness</h4><p>${(t.loudness||-12).toFixed(1)} dB</p></div>`;
}

function displayCompatibility(trackA, trackB) {
    const compatInfo = document.getElementById('compatibilityInfo');
    
    // BPM compatibility
    const bpmA = parseFloat(trackA.bpm);
    const bpmB = parseFloat(trackB.bpm);
    const bpmDiff = Math.abs(bpmA - bpmB);
    const bpmCompatible = bpmDiff < 5 || Math.abs(bpmA - bpmB/2) < 2 || Math.abs(bpmA/2 - bpmB) < 2;
    
    // Key compatibility
    const keyCompatible = checkKeyCompatibility(trackA.key_camelot, trackB.key_camelot);
    
    // Energy difference
    const energyDiff = Math.abs(parseFloat(trackA.energy_normalized) - parseFloat(trackB.energy_normalized));
    const energyCompatible = energyDiff < 0.3;
    
    compatInfo.innerHTML = `
        <div class="info-card">
            <h4>BPM Match</h4>
            <p style="color: ${bpmCompatible ? '#4caf50' : '#ff6b6b'}">
                ${bpmCompatible ? '✓ Compatible' : '✗ Adjust needed'}
            </p>
        </div>
        <div class="info-card">
            <h4>Key Match</h4>
            <p style="color: ${keyCompatible ? '#4caf50' : '#ff6b6b'}">
                ${keyCompatible ? '✓ Harmonic' : '✗ Key clash'}
            </p>
        </div>
        <div class="info-card">
            <h4>Energy Match</h4>
            <p style="color: ${energyCompatible ? '#4caf50' : '#ffa500'}">
                ${energyDiff.toFixed(2)} difference
            </p>
        </div>
        <div class="info-card">
            <h4>Overall</h4>
            <p style="color: ${(bpmCompatible && keyCompatible) ? '#4caf50' : '#ffa500'}">
                ${(bpmCompatible && keyCompatible) ? 'Great Match!' : 'Needs Work'}
            </p>
        </div>
    `;
}

function checkKeyCompatibility(keyA, keyB) {
    if (!keyA || !keyB) return false;
    
    const numA = parseInt(keyA.slice(0, -1));
    const letterA = keyA.slice(-1);
    const numB = parseInt(keyB.slice(0, -1));
    const letterB = keyB.slice(-1);
    
    // Same key
    if (keyA === keyB) return true;
    
    // Adjacent keys (same letter)
    if (letterA === letterB && Math.abs(numA - numB) === 1) return true;
    
    // Relative major/minor
    if (numA === numB && letterA !== letterB) return true;
    
    return false;
}

function displaySegments(segments) {
    if (!segments) return;
    
    const grid = document.getElementById('segmentsGrid');
    grid.innerHTML = '';
    
    segments.forEach((segment, index) => {
        const btn = document.createElement('button');
        btn.className = 'segment-btn';
        btn.innerHTML = `
            <div class="segment-label">${segment.label}</div>
            <div class="segment-time">${formatTime(segment.start)} - ${formatTime(segment.end)}</div>
        `;
        btn.onclick = () => seekToTime(segment.start);
        grid.appendChild(btn);
    });
}

// Waveform Visualization
function drawWaveform(canvasId, buffer, analysisData) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = 200;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    if (!buffer) return;
    
    // Draw segments background
    if (analysisData && analysisData.structural_analysis) {
        const duration = buffer.duration;
        
        analysisData.structural_analysis.forEach(segment => {
            const startX = (segment.start / duration) * width;
            const endX = (segment.end / duration) * width;
            
            ctx.fillStyle = getSegmentColor(segment.label);
            ctx.globalAlpha = 0.2;
            ctx.fillRect(startX, 0, endX - startX, height);
        });
        ctx.globalAlpha = 1.0;
    }
    
    // Draw waveform
    const data = buffer.getChannelData(0);
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
    
    // Draw beat markers
    if (analysisData && analysisData.beat_grid_seconds) {
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 0.5;
        ctx.globalAlpha = 0.5;
        
        const duration = buffer.duration;
        analysisData.beat_grid_seconds.forEach((beat, index) => {
            const x = (beat / duration) * width;
            
            if (index % 4 === 0) {
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.7;
            } else {
                ctx.lineWidth = 0.5;
                ctx.globalAlpha = 0.3;
            }
            
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        });
        
        ctx.globalAlpha = 1.0;
    }
}

function getSegmentColor(label) {
    const colors = {
        intro: '#4a5568',
        verse: '#2d3748',
        chorus: '#1a365d',
        breakdown: '#2c5282',
        outro: '#2b6cb0',
        main: '#1e3a5f'
    };
    return colors[label] || colors.main;
}

// Playback Controls
function togglePlayback() {
    if (isPlaying.single) {
        pausePlayback('single');
        document.getElementById('playBtn').textContent = '▶️ Play';
    } else {
        startPlayback('single');
        document.getElementById('playBtn').textContent = '⏸️ Pause';
    }
}

function togglePlaybackTrack(trackId) {
    if (isPlaying[trackId]) {
        pausePlayback(trackId);
    } else {
        startPlayback(trackId);
    }
}

function startPlayback(trackId) {
    if (!audioBuffers[trackId]) return;
    
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    audioSources[trackId] = audioContext.createBufferSource();
    audioSources[trackId].buffer = audioBuffers[trackId];
    audioSources[trackId].connect(audioContext.destination);
    
    if (pauseTimes[trackId] > 0) {
        audioSources[trackId].start(0, pauseTimes[trackId]);
        startTimes[trackId] = audioContext.currentTime - pauseTimes[trackId];
    } else {
        audioSources[trackId].start();
        startTimes[trackId] = audioContext.currentTime;
    }
    
    isPlaying[trackId] = true;
    updatePlayhead(trackId);
}

function pausePlayback(trackId) {
    if (audioSources[trackId]) {
        pauseTimes[trackId] = audioContext.currentTime - startTimes[trackId];
        audioSources[trackId].stop();
        audioSources[trackId] = null;
    }
    isPlaying[trackId] = false;
}

function stopPlayback(trackId = 'single') {
    if (audioSources[trackId]) {
        audioSources[trackId].stop();
        audioSources[trackId] = null;
    }
    isPlaying[trackId] = false;
    pauseTimes[trackId] = 0;
    updateTimeDisplay(0, trackId);
    
    // Reset playhead
    const playheadId = trackId === 'single' ? 'analysisPlayhead' : `playhead${trackId}`;
    const playhead = document.getElementById(playheadId);
    if (playhead) {
        playhead.style.left = '0px';
    }
    
    // Reset play button text
    if (trackId === 'single') {
        document.getElementById('playBtn').textContent = '▶️ Play';
    }
}

function stopPlaybackTrack(trackId) {
    stopPlayback(trackId);
}

function seekToTime(time, trackId = 'single') {
    const wasPlaying = isPlaying[trackId];
    if (isPlaying[trackId]) {
        stopPlayback(trackId);
    }
    pauseTimes[trackId] = time;
    if (wasPlaying) {
        startPlayback(trackId);
    }
}

function updatePlayhead(trackId) {
    if (!isPlaying[trackId]) return;
    
    const currentTime = audioContext.currentTime - startTimes[trackId];
    const duration = audioBuffers[trackId].duration;
    const progress = currentTime / duration;
    
    // Update playhead position
    const playheadId = trackId === 'single' ? 'analysisPlayhead' : `playhead${trackId}`;
    const playhead = document.getElementById(playheadId);
    if (playhead) {
        const canvasId = trackId === 'single' ? 'analysisWaveform' : `waveform${trackId}`;
        const canvas = document.getElementById(canvasId);
        playhead.style.left = `${progress * canvas.width}px`;
    }
    
    updateTimeDisplay(currentTime, trackId);
    
    if (currentTime < duration) {
        requestAnimationFrame(() => updatePlayhead(trackId));
    } else {
        stopPlayback(trackId);
    }
}

function updateTimeDisplay(currentTime, trackId = 'single') {
    if (trackId !== 'single') return;
    
    const duration = audioBuffers[trackId] ? audioBuffers[trackId].duration : 0;
    document.getElementById('timeDisplay').textContent = 
        `${formatTime(currentTime)} / ${formatTime(duration)}`;
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

// Enhanced compatibility renderer using API response
function getScoreColor(score) {
    if (score >= 0.8) return '#4caf50'; // green
    if (score >= 0.6) return '#ffa500'; // orange
    return '#ff6b6b';                   // red
}

function displayEnhancedCompatibility(apiData) {
    const compatInfo = document.getElementById('compatibilityInfo');

    // If API data missing or invalid fallback to basic compatibility
    if (!apiData || !apiData.compatibility) {
        displayCompatibility(analysisData.A, analysisData.B);
        return;
    }

    const comp = apiData.compatibility;
    const overall = comp.overall_score || 0;
    const bpm = comp.bpm_compatibility || {};
    const key = comp.key_compatibility || {};
    const energy = comp.energy_compatibility || {};

    compatInfo.innerHTML = `
        <div class="info-card">
            <h4>Overall</h4>
            <p style="color:${getScoreColor(overall)}">${(overall * 100).toFixed(0)}% Match</p>
        </div>
        <div class="info-card">
            <h4>BPM</h4>
            <p style="color:${getScoreColor(bpm.score || 0)}">Δ ${(bpm.difference||0).toFixed(1)} BPM</p>
        </div>
        <div class="info-card">
            <h4>Key</h4>
            <p style="color:${getScoreColor(key.score || 0)}">${key.relationship || 'N/A'}</p>
        </div>
        <div class="info-card">
            <h4>Energy</h4>
            <p style="color:${getScoreColor(energy.score || 0)}">Δ ${(energy.difference||0).toFixed(2)}</p>
        </div>
        ${comp.recommendations ? `<div class="recommendations"><h4>Tips</h4><ul>${comp.recommendations.map(r=>`<li>${r}</li>`).join('')}</ul></div>` : ''}
    `;
}

if (typeof displayAdvancedFeatures === 'undefined') {
    function displayAdvancedFeatures(features) {
        // Create section if it doesn't exist
        let advSection = document.getElementById('advancedFeatures');
        if (!advSection) {
            advSection = document.createElement('div');
            advSection.id = 'advancedFeatures';
            advSection.innerHTML = '<h3>Advanced Analysis</h3><div id="advancedFeaturesContent"></div>';
            const trackInfo = document.getElementById('trackInfo');
            if (trackInfo && trackInfo.parentNode) {
                trackInfo.parentNode.appendChild(advSection);
            }
        }
        const content = document.getElementById('advancedFeaturesContent');
        if (!content) return;

        // Build HTML from features object
        const rh = features.rhythmic || {};
        const st = features.structure || {};
        const dy = features.dynamics || {};
        content.innerHTML = `
            <div class="advanced-features-grid">
                <div class="feature-group">
                    <h4>Rhythmic</h4>
                    <p>Tempo Conf.: ${(rh.tempo_confidence*100||0).toFixed(0)}%</p>
                    <p>Beat Count: ${rh.beat_count||0}</p>
                </div>
                <div class="feature-group">
                    <h4>Structure</h4>
                    <p>Sections: ${st.section_count||0}</p>
                    <p>Complexity: ${st.structure_complexity||0}</p>
                </div>
                <div class="feature-group">
                    <h4>Dynamics</h4>
                    <p>Range: ${dy.loudness_range?.toFixed? dy.loudness_range.toFixed(1):'N/A'} dB</p>
                    <p>Peak: ${dy.peak_loudness?.toFixed? dy.peak_loudness.toFixed(1):'N/A'} dB</p>
                </div>
            </div>`;
    }
}
