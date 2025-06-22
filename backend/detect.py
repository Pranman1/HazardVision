import cv2
import os
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from utils import save_snapshot, log_hazard
import glob
from pathlib import Path

# Check for CUDA availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLOv8 model with CUDA support and better confidence threshold
model = YOLO("yolov8m.pt")
model.to(device)  # Move model to GPU
conf_threshold = 0.3  # Lower confidence threshold for better knife detection

# Enhanced OSHA-aligned hazard categories
HAZARD_CATEGORIES = {
    "fall_hazards": {
        "objects": {"ladder", "chair", "platform", "scaffold", "stairs"},
        "min_height": 4  # feet - OSHA requires fall protection above 4 feet
    },
    "trip_hazards": {
        "objects": {"bottle", "cord", "box", "suitcase", "backpack", "bag", "wire", "cable"},
        "ground_level": True  # Objects at ground level are trip hazards
    },
    "sharp_objects": {
        "objects": {"knife", "scissors", "tool", "saw", "drill", "blade", "cutter"},
        "confidence_threshold": 0.3,  # Lower threshold specifically for sharp objects
        "safe_handling": {
            "required_context": ["hand"],  # Objects that need proper handling context
            "safe_distance": 50  # pixels - safe handling distance threshold
        }
    },
    "fire_hazards": {
        "objects": {"fire", "smoke", "cigarette", "matches", "lighter"},
        "always_hazard": True  # These are always considered hazardous
    },
    "electrical_hazards": {
        "objects": {"cord", "wire", "cable", "outlet", "power strip", "electrical panel"},
        "context_sensitive": True
    },
    "chemical_hazards": {
        "objects": {"bottle", "container", "spray", "tank"},
        "context_sensitive": True
    },
    "obstacles": {
        "objects": set(),  # Will be populated with any object in walkways
        "proximity_threshold": 100  # pixels - distance to consider as obstacle
    }
}

# Enhanced hazardous object combinations
HAZARD_COMBINATIONS = [
    # Fall hazards - person on elevated surface
    {"person", "ladder"},
    {"person", "chair"},
    {"person", "platform"},
    {"person", "scaffold"},
    {"person", "stairs"},
    
    # Trip hazards - objects in walkways
    {"person", "cord"},
    {"person", "wire"},
    {"person", "cable"},
    {"person", "bottle"},
    {"person", "box"},
    
    # Sharp object hazards
    {"person", "knife"},
    {"person", "scissors"},
    {"person", "saw"},
    {"person", "drill"},
    
    # Fire hazards
    {"fire"},
    {"smoke"},
    {"person", "cigarette"},
    
    # Electrical hazards
    {"person", "cord"},
    {"person", "wire"},
    {"person", "cable"},
    {"person", "outlet"},
    {"person", "electrical panel"},
    
    # Multiple people in hazardous situation
    {"person", "person", "ladder"},
    {"person", "person", "scaffold"},
]

def is_object_held_safely(box1, box2, safe_distance=50):
    """
    Determine if an object is being held safely by checking spatial relationship
    between the hand/person and the object
    """
    x1_hand = (box1["box"][0] + box1["box"][2]) / 2
    y1_hand = (box1["box"][1] + box1["box"][3]) / 2
    x2_obj = (box2["box"][0] + box2["box"][2]) / 2
    y2_obj = (box2["box"][1] + box2["box"][3]) / 2
    
    # Calculate distance between centers
    distance = np.sqrt((x1_hand - x2_obj)**2 + (y1_hand - y2_obj)**2)
    
    # Check if hand is near object
    return distance <= safe_distance

def analyze_spatial_relationships(boxes):
    """
    Analyze spatial relationships between detected objects
    Returns dictionary of spatial hazards found
    """
    spatial_hazards = []
    
    try:
        # Analyze each pair of boxes
        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes[i+1:], start=i+1):
                # Check for sharp objects
                if box1["label"] in HAZARD_CATEGORIES["sharp_objects"]["objects"]:
                    # Look for hands or people nearby
                    if box2["label"] in ["hand", "person"]:
                        if is_object_held_safely(box2, box1):
                            # Object is being held properly
                            continue
                        else:
                            spatial_hazards.append({
                                "type": "unsafe_sharp_object_handling",
                                "objects": [box1["label"], box2["label"]],
                                "severity": "high"
                            })
                    else:
                        # Sharp object without proper handling context
                        spatial_hazards.append({
                            "type": "unattended_sharp_object",
                            "objects": [box1["label"]],
                            "severity": "medium"
                        })
                
                # Check for trip hazards near people
                if (box1["label"] == "person" and 
                    box2["label"] in HAZARD_CATEGORIES["trip_hazards"]["objects"]):
                    if is_near_feet(box1, box2):
                        spatial_hazards.append({
                            "type": "trip_hazard",
                            "objects": [box1["label"], box2["label"]],
                            "severity": "medium"
                        })
                
                # Check for electrical hazards
                if (box1["label"] in HAZARD_CATEGORIES["electrical_hazards"]["objects"] and
                    box2["label"] == "person"):
                    if objects_are_close(box1, box2):
                        spatial_hazards.append({
                            "type": "electrical_hazard",
                            "objects": [box1["label"], box2["label"]],
                            "severity": "high"
                        })
                        
    except Exception as e:
        print(f"Error in spatial analysis: {str(e)}")
        return []
        
    return spatial_hazards

def is_near_feet(person_box, object_box, threshold=50):
    """Check if an object is near a person's feet"""
    person_feet = person_box["box"][3]  # Bottom of person box
    object_top = object_box["box"][1]   # Top of object box
    return abs(person_feet - object_top) < threshold

def objects_are_close(box1, box2, threshold=100):
    """Check if two objects are close to each other"""
    x1_center = (box1["box"][0] + box1["box"][2]) / 2
    y1_center = (box1["box"][1] + box1["box"][3]) / 2
    x2_center = (box2["box"][0] + box2["box"][2]) / 2
    y2_center = (box2["box"][1] + box2["box"][3]) / 2
    
    distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    return distance < threshold

def classify_hazard(labels, boxes):
    """
    Enhanced hazard classification based on OSHA guidelines and spatial analysis.
    Returns tuple of (is_hazardous, hazard_types, severity)
    """
    label_set = set(labels)
    hazard_types = []
    severity = "low"
    is_hazardous = False  # Initialize is_hazardous at the start
    
    try:
        # First check for inherently hazardous objects
        for category, rules in HAZARD_CATEGORIES.items():
            if rules.get("always_hazard", False):
                if any(obj in label_set for obj in rules["objects"]):
                    hazard_types.append(category)
                    severity = "high"
                    is_hazardous = True
        
        # Analyze spatial relationships for context-sensitive hazards
        spatial_hazards = analyze_spatial_relationships(boxes)
        
        # Process spatial hazards
        for hazard in spatial_hazards:
            if hazard["type"] not in hazard_types:
                hazard_types.append(hazard["type"])
            if hazard["severity"] == "high":
                severity = "high"
            elif hazard["severity"] == "medium" and severity != "high":
                severity = "medium"
            is_hazardous = True
        
        # Check for unhandled sharp objects
        for label in labels:
            if label in HAZARD_CATEGORIES["sharp_objects"]["objects"]:
                # Check if there's a hand nearby in the boxes
                has_proper_handling = False
                for box in boxes:
                    if box["label"] == "hand":
                        for other_box in boxes:
                            if other_box["label"] == label:
                                if is_object_held_safely(box, other_box):
                                    has_proper_handling = True
                                    break
                
                if not has_proper_handling:
                    hazard_types.append("unattended_sharp_object")
                    severity = max(severity, "medium")
                    is_hazardous = True
        
        # Check for obstacles in walkways
        for box in boxes:
            if (box["label"] not in ["wall", "floor", "ceiling"] and 
                box["label"] not in [h["objects"][0] for h in hazard_types]):
                # Add as general obstacle if not already classified as another hazard
                hazard_types.append({
                    "type": "obstacle",
                    "objects": [box["label"]],
                    "severity": "low"
                })
                is_hazardous = True
                
    except Exception as e:
        print(f"Error in hazard classification: {str(e)}")
        # Return safe defaults in case of error
        return False, [], "low"
    
    return is_hazardous, hazard_types, severity

def cleanup_old_snapshots(snapshots_dir, max_files=50):
    """Keep only the most recent max_files in the snapshots directory"""
    try:
        files = glob.glob(os.path.join(snapshots_dir, "hazard_*.jpg"))
        if len(files) > max_files:
            # Sort files by modification time
            files.sort(key=os.path.getmtime)
            # Remove oldest files
            for f in files[:-max_files]:
                try:
                    os.remove(f)
                except:
                    pass
    except Exception as e:
        print(f"Error cleaning snapshots: {str(e)}")

def detect_and_log(image_path, timestamp):
    """Run object detection and return results with bounding boxes."""
    try:
        # Run detection with CUDA and adjusted confidence
        results = model(image_path, conf=conf_threshold, device=device)
        labels = []
        boxes = []
        
        # Process results without saving intermediate images
        for result in results:
            for box in result.boxes:
                # Get confidence score
                conf = float(box.conf)
                # Get label
                label = result.names[int(box.cls)]
                
                # Use lower threshold for sharp objects
                if label in HAZARD_CATEGORIES["sharp_objects"]["objects"]:
                    min_conf = HAZARD_CATEGORIES["sharp_objects"]["confidence_threshold"]
                else:
                    min_conf = conf_threshold
                
                if conf >= min_conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "label": label,
                        "confidence": conf
                    })
                    labels.append(label)
        
        # Classify hazards
        is_hazardous, hazard_types, severity = classify_hazard(labels, boxes)
        
        # Only save image if there's a hazard
        if is_hazardous:
            img = cv2.imread(image_path)
            if img is not None:
                # Draw boxes only for hazardous objects
                for box in boxes:
                    if (box["label"] in [item for sublist in [cat["objects"] for cat in HAZARD_CATEGORIES.values()] for item in sublist]):
                        x1, y1, x2, y2 = box["box"]
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(img, f"{box['label']}: {box['confidence']:.2f}", 
                                  (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 255), 2)
                
                # Save with reduced quality for smaller file size
                output_path = os.path.join("snapshots", f"hazard_{timestamp}.jpg")
                cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Keep only last 50 hazard images
                cleanup_old_snapshots(os.path.dirname(output_path), max_files=50)
        
        return {
            "is_hazardous": is_hazardous,
            "hazard_types": hazard_types,
            "severity": severity,
            "boxes": boxes,
            "labels": labels
        }
        
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        return {
            "is_hazardous": False,
            "hazard_types": [],
            "severity": "low",
            "boxes": [],
            "labels": []
        }