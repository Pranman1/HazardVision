import json
import os
from datetime import datetime
import cv2

def log_hazard(labels, timestamp, image_path):
    """Log hazard detection to file - minimal logging"""
    try:
        log_entry = {
            "timestamp": timestamp,
            "labels": labels,
            "image": os.path.basename(image_path)
        }
        
        log_file = "hazard_log.json"
        
        # Keep log file small by limiting entries
        entries = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    entries = json.load(f)
                    # Keep only last 100 entries
                    entries = entries[-100:]
            except:
                entries = []
        
        entries.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(entries[-100:], f)  # Only keep last 100 entries
            
    except Exception as e:
        print(f"Error logging hazard: {str(e)}")

def save_snapshot(frame, filename):
    """Save a frame to disk with compression"""
    try:
        # Save with reduced quality
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True
    except Exception as e:
        print(f"Error saving snapshot: {str(e)}")
        return False
