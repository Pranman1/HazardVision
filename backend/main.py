from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import base64
import numpy as np
import json
from threading import Thread, Lock
from queue import Queue
import asyncio
from datetime import datetime
import torch
from detect import HAZARD_CATEGORIES, classify_hazard
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")
app.mount("/snapshots", StaticFiles(directory="../snapshots"), name="snapshots")

# Initialize model with GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YOLO("yolov8m.pt")
model.to(device)

# Initialize GPU memory management
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True

# Shared resources
frame_queue = Queue(maxsize=2)  # Only keep latest frame
result_lock = Lock()
latest_result = None

def process_frame(frame):
    """Process a single frame with the model"""
    try:
        # Resize frame for faster processing while maintaining aspect ratio
        max_size = 640
        height, width = frame.shape[:2]
        scale = min(max_size/width, max_size/height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame

        # Run detection on GPU with mixed precision
        with torch.cuda.amp.autocast():
            results = model(frame_resized)

        labels = []
        boxes = []
        
        # Process results
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                conf = float(box.conf)
                
                # Scale coordinates back to original size if resized
                if scale < 1:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 / scale)
                    x2 = int(x2 / scale)
                    y1 = int(y1 / scale)
                    y2 = int(y2 / scale)
                else:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                labels.append(label)
                boxes.append({
                    "label": label,
                    "confidence": float(conf),
                    "box": [x1, y1, x2, y2]
                })

                # Draw box and label
                color = (0, 255, 0)  # Default green
                for category, rules in HAZARD_CATEGORIES.items():
                    if label in rules["objects"]:
                        if "fall" in category or "sharp" in category or "fire" in category:
                            color = (0, 0, 255)  # Red for serious hazards
                        elif "trip" in category:
                            color = (255, 165, 0)  # Orange for trip hazards
                        elif "electrical" in category:
                            color = (255, 255, 0)  # Yellow for electrical
                        break

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Classify hazard
        is_hazardous, hazard_types, severity = classify_hazard(labels, boxes)
        
        # Convert frame to base64 for sending
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "frame": frame_b64,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "labels": labels,
            "boxes": boxes,
            "is_hazardous": is_hazardous,
            "hazard_types": hazard_types,
            "severity": severity
        }

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

def detection_thread():
    """Background thread for processing frames"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                result = process_frame(frame)
                if result:
                    with result_lock:
                        global latest_result
                        latest_result = result
            except Exception as e:
                logger.error(f"Error in detection thread: {str(e)}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Start detection thread
Thread(target=detection_thread, daemon=True).start()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        while True:
            try:
                # Receive frame data
                data = await websocket.receive_text()
                
                # Parse frame data
                frame_data = json.loads(data)
                if 'frame' not in frame_data:
                    continue
                
                # Decode frame
                frame_bytes = base64.b64decode(frame_data['frame'].split(',')[1])
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Update frame queue
                    if frame_queue.full():
                        frame_queue.get()  # Remove old frame if queue is full
                    frame_queue.put(frame)
                    
                    # Get latest result
                    with result_lock:
                        result = latest_result
                    
                    if result:
                        await websocket.send_json(result)
                
            except WebSocketDisconnect:
                logger.info("Client disconnected normally")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")

# Keep the old snapshot endpoint for backup
@app.post("/snapshot/")
async def process_snapshot():
    """Legacy endpoint - kept for backup"""
    pass  # Implementation removed as we're using WebSocket now
