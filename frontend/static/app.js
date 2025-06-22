const video = document.getElementById('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
const table = document.getElementById('logTable');
let isRunning = false;
let ws = null;

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = Date.now();
const fpsDisplay = document.getElementById('fpsDisplay');
let skipFrames = 2; // Process every 3rd frame
let frameSkipCount = 0;

// Detection settings
const RECONNECT_TIMEOUT = 3000; // 3 seconds

// Audio feedback for hazards
let audioContext = null;
let lastAudioTime = 0;
const AUDIO_COOLDOWN = 3000; // 3 seconds between alerts

// Audio handling
let currentAudio = null;

function createAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
}

function playHazardAlert(severity) {
    const now = Date.now();
    if (now - lastAudioTime > AUDIO_COOLDOWN) {
        const context = createAudioContext();
        
        // Create oscillator for beep
        const oscillator = context.createOscillator();
        const gainNode = context.createGain();
        
        // Set frequency based on severity
        switch(severity) {
            case 'high':
                oscillator.frequency.setValueAtTime(880, context.currentTime); // A5
                break;
            case 'medium':
                oscillator.frequency.setValueAtTime(440, context.currentTime); // A4
                break;
            default:
                oscillator.frequency.setValueAtTime(220, context.currentTime); // A3
        }
        
        // Connect nodes
        oscillator.connect(gainNode);
        gainNode.connect(context.destination);
        
        // Set volume envelope
        gainNode.gain.setValueAtTime(0, context.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.5, context.currentTime + 0.1);
        gainNode.gain.linearRampToValueAtTime(0, context.currentTime + 0.5);
        
        // Start and stop
        oscillator.start(context.currentTime);
        oscillator.stop(context.currentTime + 0.5);
        
        lastAudioTime = now;
    }
}

function playHazardAudio(audioPath) {
    try {
        // Stop any currently playing audio
        if (currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }
        
        // Create and play new audio
        currentAudio = new Audio(audioPath);
        currentAudio.play();
    } catch (error) {
        console.error('Error playing audio:', error);
    }
}

// Helper function to create image elements with proper loading states
function createImage(src, width = null) {
    const img = document.createElement("img");
    if (width) img.width = width;
    img.alt = "Loading...";
    img.classList.add("loading");
    
    img.onload = function() {
        img.classList.remove("loading");
        img.classList.remove("error");
        img.alt = "Detection result";
    };
    
    img.onerror = function() {
        console.error(`Failed to load image: ${src}`);
        img.classList.remove("loading");
        img.classList.add("error");
        img.alt = "Failed to load image";
        img.style.backgroundColor = "#f8d7da";
        img.style.border = "1px solid #f5c6cb";
    };
    
    img.src = src;
    return img;
}

// Start camera
async function initCamera() {
    try {
        // First enumerate devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        // Try to find DroidCam
        const droidcam = videoDevices.find(device => device.label.toLowerCase().includes('droidcam'));
        
        // Get video stream
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 },
                deviceId: droidcam ? droidcam.deviceId : undefined
            }
        });
        
        video.srcObject = stream;
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        });
        
        console.log('Camera initialized successfully');
        console.log('Available video devices:', videoDevices.map(d => d.label));
        
    } catch (err) {
        console.error("Camera error:", err);
    }
}

// Initialize camera
initCamera();

function connectWebSocket() {
    if (ws) {
        ws.close();
    }
    
    ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        if (isRunning) {
            startDetection();
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        setTimeout(connectWebSocket, RECONNECT_TIMEOUT);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Update hazard info display
        updateHazardInfo(data);
        
        // Update detection image
        const annotatedImg = document.getElementById('annotatedImg');
        if (!annotatedImg) return;
        
        // Create new image from base64 data
        annotatedImg.src = 'data:image/jpeg;base64,' + data.frame;
        
        // Play alert if hazardous
        if (data.is_hazardous) {
            if (data.severity === 'critical') {
                // Play beep alert
                playHazardAlert(data.severity);
                
                // If we have a hazard analysis with audio, play it
                if (data.hazard_analysis && data.hazard_analysis.audio_path) {
                    playHazardAudio(data.hazard_analysis.audio_path);
                }
            }
            
            // Add to log
            const table = document.getElementById('logTable');
            if (!table) return;

            const tbody = table.querySelector('tbody') || table;
            const row = tbody.insertRow(0);
            row.insertCell().innerText = data.timestamp;
            row.insertCell().innerText = data.labels.join(", ");
            
            // Add analysis if available
            const hazardCell = row.insertCell();
            if (data.hazard_analysis && data.hazard_analysis.analysis) {
                hazardCell.innerText = data.hazard_analysis.analysis;
            } else {
                hazardCell.innerText = data.hazard_types.join(", ");
            }
            
            const severityCell = row.insertCell();
            severityCell.innerHTML = `<span class="severity-${data.severity}">${data.severity.toUpperCase()}</span>`;
            
            const imgCell = row.insertCell();
            const img = document.createElement('img');
            img.src = 'data:image/jpeg;base64,' + data.frame;
            img.width = 100;
            imgCell.appendChild(img);
        }
        
        // Update FPS
        updateFPS();
    };
}

// Update FPS display
function updateFPS() {
    frameCount++;
    const now = Date.now();
    const elapsed = now - lastFpsUpdate;
    
    if (elapsed >= 1000) { // Update every second
        const fps = Math.round((frameCount * 1000) / elapsed);
        fpsDisplay.textContent = `${fps} FPS`;
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

function startDetection() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        return;
    }
    
    // Process frame
    ctx.drawImage(video, 0, 0);
    
    // Skip frames to reduce load
    if (frameSkipCount++ % skipFrames === 0) {
        // Convert to base64
        const frame = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server
        ws.send(JSON.stringify({ frame }));
    }
    
    if (isRunning) {
        requestAnimationFrame(startDetection);
    }
}

// Toggle detection
document.getElementById('toggleBtn').onclick = () => {
    isRunning = !isRunning;
    document.getElementById('toggleBtn').textContent = isRunning ? 'Stop Detection' : 'Start Detection';
    
    if (isRunning) {
        // Reset FPS counter
        frameCount = 0;
        lastFpsUpdate = Date.now();
        
        // Start detection
        startDetection();
    } else {
        // Clear detection view
        document.getElementById('annotatedImg').src = '';
        document.getElementById('hazardInfo').innerHTML = '';
        fpsDisplay.textContent = '';
        
        // Close WebSocket
        if (ws) {
            ws.close();
        }
    }
};

// Update hazard info display
function updateHazardInfo(data) {
    const hazardInfo = document.getElementById('hazardInfo');
    if (!hazardInfo) return;

    // Clear previous content
    hazardInfo.innerHTML = '';

    // Create hazard types and severity divs
    const hazardTypes = document.createElement('div');
    hazardTypes.id = 'hazardTypes';
    const hazardSeverity = document.createElement('div');
    hazardSeverity.id = 'hazardSeverity';
    
    hazardInfo.appendChild(hazardTypes);
    hazardInfo.appendChild(hazardSeverity);

    // Remove previous severity class
    hazardInfo.classList.remove('high', 'medium', 'low');

    if (data.is_hazardous) {
        // Add severity class
        hazardInfo.classList.add(data.severity);
        
        // Show hazard types
        hazardTypes.innerHTML = `<strong>Hazard Types:</strong> ${data.hazard_types.join(', ')}`;
        
        // Show severity
        hazardSeverity.innerHTML = `<strong>Severity:</strong> <span class="severity-${data.severity}">${data.severity.toUpperCase()}</span>`;
    } else {
        hazardTypes.innerHTML = '<strong>No hazards detected</strong>';
    }
}
