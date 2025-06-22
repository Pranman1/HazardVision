import json
import os

def log_hazard(objects, timestamp, img_path):
    """Log a hazard detection with proper relative paths."""
    # Convert to relative path for storage
    relative_path = f"{timestamp}.jpg"
    
    entry = {
        "timestamp": timestamp,
        "objects": objects,
        "snapshot": f"/snapshots/{relative_path}"
    }
    
    log_file = os.path.join(os.path.dirname(__file__), "hazard_log.json")
    try:
        with open(log_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
        
    data.append(entry)
    
    with open(log_file, "w") as f:
        json.dump(data, f, indent=4)

def save_snapshot():
    pass  # handled directly in main.py
