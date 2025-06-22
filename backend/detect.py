import cv2
import os
import torch
from datetime import datetime
from ultralytics import YOLO
from utils import save_snapshot, log_hazard
import glob
from pathlib import Path

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# OSHA-aligned hazard categories
HAZARD_CATEGORIES = {
    "fall_hazards": {
        "objects": {"ladder", "chair"},  # Things people might climb on unsafely
        "min_height": 4  # feet - OSHA requires fall protection above 4 feet in general industry
    },
    "trip_hazards": {
        "objects": {"bottle", "cord", "box", "suitcase"}  # Objects that could cause trips
    },
    "sharp_objects": {
        "objects": {"knife", "scissors"}
    },
    "fire_hazards": {
        "objects": {"fire", "smoke"}
    },
    "electrical_hazards": {
        "objects": {"cord", "wire"}
    },
    "chemical_hazards": {
        "objects": {"bottle"}  # Could contain hazardous materials
    }
}

# Define hazardous object combinations based on OSHA standards
HAZARD_COMBINATIONS = [
    # Fall hazards - person on elevated surface
    {"person", "ladder"},  # Person on ladder
    {"person", "chair"},   # Person standing on chair (unsafe)
    
    # Trip hazards - objects in walkways
    {"person", "cord"},    # Person near cord (trip hazard)
    {"person", "bottle"},  # Person near obstacle
    
    # Sharp object hazards
    {"person", "knife"},   # Person handling sharp object
    {"person", "scissors"},# Person handling sharp object
    
    # Fire hazards
    {"fire"},             # Any fire is hazardous
    {"smoke"},            # Smoke indicates potential fire
    
    # Electrical hazards
    {"person", "cord"},   # Person near exposed electrical
    {"person", "wire"},   # Person near exposed wiring
    
    # Multiple people in hazardous situation
    {"person", "person", "ladder"},  # Multiple people on/near ladder
]

def classify_hazard(labels, boxes):
    """
    Classify the type and severity of hazard based on OSHA guidelines.
    Returns tuple of (is_hazardous, hazard_types, severity)
    """
    label_set = set(labels)
    hazard_types = []
    severity = "low"
    
    # Check each hazard category
    for category, rules in HAZARD_CATEGORIES.items():
        if any(obj in label_set for obj in rules["objects"]):
            hazard_types.append(category)
    
    # Check for hazardous combinations
    is_hazardous = False
    for combo in HAZARD_COMBINATIONS:
        if combo.issubset(label_set):
            is_hazardous = True
            severity = "high" if len(combo) > 2 else "medium"
            
    return is_hazardous, hazard_types, severity

def cleanup_old_snapshots(snapshots_dir, max_files=100):
    """Clean up old snapshot files to prevent disk space issues"""
    try:
        files = glob.glob(os.path.join(snapshots_dir, "*.jpg"))
        if len(files) > max_files:
            # Sort by modification time and remove oldest
            files.sort(key=lambda x: os.path.getmtime(x))
            for f in files[:-max_files]:
                try:
                    os.remove(f)
                except OSError:
                    pass  # Ignore errors during cleanup
    except Exception as e:
        print(f"Error during cleanup: {e}")

def detect_and_log(image_path, timestamp):
    """Run object detection and return results with bounding boxes."""
    # Run detection
    results = model(image_path)
    labels = []
    boxes = []
    
    # Get original image for drawing
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image")
    
    # Create output path - use the same filename as input to avoid confusion
    output_path = image_path
    
    for result in results:
        for box in result.boxes:
            # Get label
            label = result.names[int(box.cls)]
            labels.append(label)
            
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            
            boxes.append({
                "label": label,
                "confidence": conf,
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            # Determine color based on hazard category
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
            
            # Draw box and label with improved visibility
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Add background to text for better visibility
            label_text = f"{label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (int(x1), int(y1)-text_height-10), (int(x1)+text_width, int(y1)), color, -1)
            cv2.putText(img, label_text, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save annotated image
    cv2.imwrite(output_path, img)
    
    # Cleanup old files (keep more recent files to avoid issues)
    cleanup_old_snapshots(os.path.dirname(output_path), max_files=200)
    
    # Classify hazard
    is_hazardous, hazard_types, severity = classify_hazard(labels, boxes)
    
    # Only log if hazardous
    if is_hazardous:
        log_hazard(labels, timestamp, output_path)
    
    return {
        "timestamp": timestamp,  # Return original timestamp
        "labels": labels,
        "boxes": boxes,
        "is_hazardous": is_hazardous,
        "hazard_types": hazard_types,
        "severity": severity
    }