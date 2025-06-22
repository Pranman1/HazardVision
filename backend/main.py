from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from detect import detect_and_log
import os
from datetime import datetime
import json

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static folder to serve snapshots - use absolute path but store relative paths
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
snapshot_dir = os.path.join(WORKSPACE_ROOT, "snapshots")
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

# Mount snapshots directory
app.mount("/snapshots", StaticFiles(directory=snapshot_dir), name="snapshots")

@app.post("/snapshot/")
async def snapshot(file: UploadFile = File(...)):
    """Process a single frame from the webcam feed."""
    # Generate unique timestamp for this frame
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    
    # Save uploaded image with relative path for storage
    filename = f"{timestamp}.jpg"
    save_path = os.path.join(snapshot_dir, filename)
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the file
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # Run detection and get results
        results = detect_and_log(save_path, timestamp)
        
        return {
            "timestamp": timestamp,
            "labels": results["labels"],
            "boxes": results["boxes"],
            "is_hazardous": results["is_hazardous"],
            "hazard_types": results.get("hazard_types", []),
            "severity": results.get("severity", "low")
        }
    except Exception as e:
        # If there's an error, try to remove the failed file
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except:
                pass
        raise Exception(f"Error processing snapshot: {str(e)}")

@app.get("/log/")
def get_log():
    """Get the hazard detection log."""
    try:
        with open("hazard_log.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Initialize hazard log if it doesn't exist
log_file = os.path.join(os.path.dirname(__file__), "hazard_log.json")
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        json.dump([], f)
