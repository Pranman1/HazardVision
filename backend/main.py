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
from hazard_analysis import process_critical_hazard
import os

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

# Model configuration
conf_threshold = 0.3  # Confidence threshold for detections

def process_frame(frame):
    """Process a single frame with the model"""
    try:
        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = model(frame_rgb, conf=conf_threshold)
        
        labels = []
        boxes = []
        
        # Process results
        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                label = result.names[int(box.cls)]
                
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "label": label,
                        "confidence": conf
                    })
                    labels.append(label)
                
                # Default green
                color = (0, 255, 0)
                for category, rules in HAZARD_CATEGORIES.items():
                    if label in rules["objects"]:
                        if "fall" in category or "sharp" in category or "fire" in category:
                            color = (0, 0, 255)  # Red for serious hazards
                        elif "trip" in category:
                            color = (255, 165, 0)  # Orange for trip hazards
                        elif "electrical" in category:
                            color = (255, 255, 0)  # Yellow for electrical
                        break

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(frame, label_text, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Classify hazard
        is_hazardous, hazard_types, severity = classify_hazard(labels, boxes)
        
        # Process critical hazards with LLM and speech synthesis
        hazard_analysis = None
        if is_hazardous and severity == "critical":
            # Save frame for analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = f"snapshots/hazard_{timestamp}.jpg"
            cv2.imwrite(frame_path, frame)
            
            # Get hazard analysis
            hazard_analysis = process_critical_hazard(frame_path, hazard_types, severity)
        
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
            "severity": severity,
            "hazard_analysis": hazard_analysis
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
                        try:
                            frame_queue.get_nowait()  # Remove old frame if queue is full
                        except:
                            pass
                    try:
                        frame_queue.put_nowait(frame)
                    except:
                        pass
                    
                    # Get latest result
                    with result_lock:
                        result = latest_result
                    
                    if result:
                        await websocket.send_json(result)
                
            except WebSocketDisconnect:
                logger.info("Client disconnected normally")
                break
            except asyncio.CancelledError:
                logger.info("WebSocket connection cancelled")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up resources
        logger.info("WebSocket connection closed")
        try:
            # Clear frame queue
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
        except:
            pass

# Keep the old snapshot endpoint for backup
@app.post("/snapshot/")
async def process_snapshot():
    """Legacy endpoint - kept for backup"""
    pass  # Implementation removed as we're using WebSocket now
