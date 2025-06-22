const video = document.getElementById('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
const table = document.getElementById('logTable');
let isRunning = false;
let detectionInterval = null;

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = Date.now();
const fpsDisplay = document.getElementById('fpsDisplay');

// Detection settings
const DETECTION_INTERVAL = 100; // 10 fps (1000ms / 100ms = 10) - more reasonable for processing
const MAX_RETRIES = 3;
let lastImageTimestamp = null;

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
        
        // Just show error state instead of trying to load an error image
        img.style.backgroundColor = "#f8d7da";
        img.style.border = "1px solid #f5c6cb";
    };
    
    img.src = src;
    return img;
}

// Start camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        });
    })
    .catch(err => console.error("Camera error:", err));

// Toggle detection
document.getElementById('toggleBtn').onclick = () => {
    isRunning = !isRunning;
    document.getElementById('toggleBtn').textContent = isRunning ? 'Stop Detection' : 'Start Detection';
    
    if (isRunning) {
        // Reset FPS counter
        frameCount = 0;
        lastFpsUpdate = Date.now();
        
        // Start detection loop
        detectionInterval = setInterval(detectFrame, DETECTION_INTERVAL);
    } else {
        // Stop detection loop
        if (detectionInterval) {
            clearInterval(detectionInterval);
            detectionInterval = null;
        }
        // Clear FPS display
        fpsDisplay.textContent = '';
        // Clear detection view
        document.getElementById('annotatedImg').src = '';
        document.getElementById('hazardInfo').innerHTML = '';
    }
};

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

// Update hazard info display
function updateHazardInfo(data) {
    const hazardInfo = document.getElementById('hazardInfo');
    if (!hazardInfo) return;  // Guard against missing element

    // Clear previous content
    hazardInfo.innerHTML = '';

    // Create hazard types and severity divs if they don't exist
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

// Process a single frame
async function detectFrame() {
    if (!isRunning) return;

    // Update FPS counter
    updateFPS();

    try {
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0);
        
        // Convert to blob and send to backend
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));
        const form = new FormData();
        form.append("file", blob, "snapshot.jpg");

        const res = await fetch("http://localhost:8000/snapshot/", {
            method: "POST",
            body: form,
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        });

        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        
        // Check if this is a new frame
        if (data.timestamp === lastImageTimestamp) {
            console.log("Received duplicate frame, skipping update");
            return;
        }
        lastImageTimestamp = data.timestamp;
        
        // Update hazard info display
        updateHazardInfo(data);

        // Create image URL with stronger cache-busting
        const cacheBuster = Date.now() + Math.random();
        const imageUrl = `http://localhost:8000/snapshots/${data.timestamp}.jpg?nocache=${cacheBuster}`;

        // Update live view with annotated image
        const annotatedImg = document.getElementById('annotatedImg');
        if (!annotatedImg) return;  // Guard against missing element

        // Create new image and wait for it to load before replacing
        const newImg = createImage(imageUrl);
        newImg.id = 'annotatedImg';
        
        // Only replace once loaded
        newImg.onload = function() {
            annotatedImg.parentNode.replaceChild(newImg, annotatedImg);
        };

        // Handle load failure with retries
        let retryCount = 0;
        newImg.onerror = async function() {
            if (retryCount < MAX_RETRIES) {
                retryCount++;
                console.log(`Retrying image load (${retryCount}/${MAX_RETRIES})`);
                // Add random delay before retry
                await new Promise(resolve => setTimeout(resolve, 100 * retryCount));
                // Try with both original and annotated filenames
                const retryUrl = retryCount % 2 === 0 ? 
                    `http://localhost:8000/snapshots/${data.timestamp}.jpg?retry=${retryCount}` :
                    `http://localhost:8000/snapshots/${data.timestamp}_annotated.jpg?retry=${retryCount}`;
                newImg.src = retryUrl;
            } else {
                console.error(`Failed to load image after ${MAX_RETRIES} retries`);
                newImg.classList.add('error');
                newImg.alt = 'Detection failed';
            }
        };
        
        // Only add to log if hazards detected
        if (data.is_hazardous) {
            const table = document.getElementById('logTable');
            if (!table) return;  // Guard against missing element

            const tbody = table.querySelector('tbody') || table;
            const row = tbody.insertRow(0);  // Insert at top of table
            row.insertCell().innerText = data.timestamp;
            row.insertCell().innerText = data.labels.join(", ");
            row.insertCell().innerText = data.hazard_types.join(", ");
            
            const severityCell = row.insertCell();
            severityCell.innerHTML = `<span class="severity-${data.severity}">${data.severity.toUpperCase()}</span>`;
            
            const imgCell = row.insertCell();
            const img = createImage(imageUrl, 100);
            imgCell.appendChild(img);
        }

    } catch (err) {
        console.error("Detection error:", err);
        
        // Show error in hazard info
        const hazardInfo = document.getElementById('hazardInfo');
        if (hazardInfo) {
            hazardInfo.innerHTML = `<div class="error">Detection Error: ${err.message}</div>`;
        }
        
        // Clear or show error state for annotated image
        const annotatedImg = document.getElementById('annotatedImg');
        if (annotatedImg) {
            annotatedImg.classList.add('error');
            annotatedImg.alt = 'Detection failed';
        }
    }
}

// Add error handling for annotated image
document.getElementById('annotatedImg').onerror = function() {
  console.error(`Failed to load annotated image: ${this.src}`);
  this.src = '/static/error-image.png';  // Fallback image
  this.alt = 'Detection image failed to load';
};
