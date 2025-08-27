import cv2
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from utils import save_snapshot, log_hazard
import glob
from pathlib import Path
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "min_height": 4,  # feet - OSHA requires fall protection above 4 feet
        "severity": "high"
    },
    "floor_obstacles": {  # New category specifically for floor hazards
        "objects": {"bag", "backpack", "suitcase", "box", "cord", "cable", "bottle"},
        "severity": "critical",  # Increased severity for floor obstacles
        "ground_level": True
    },
    "sharp_objects": {
        "objects": {"knife", "scissors", "tool", "saw", "drill", "blade", "cutter"},
        "confidence_threshold": 0.3,
        "safe_handling": {
            "required_context": ["hand", "person"],
            "safe_distance": 50,
            "severity": {
                "unattended": "critical",
                "improper_handling": "high",
                "proper_handling": "medium"
            }
        }
    },
    "fire_hazards": {
        "objects": {"fire", "smoke", "cigarette", "matches", "lighter"},
        "always_hazard": True,
        "severity": "critical"
    },
    "electrical_hazards": {
        "objects": {"cord", "wire", "cable", "outlet", "power strip", "electrical panel"},
        "context_sensitive": True,
        "severity": "high"
    },
    "chemical_hazards": {
        "objects": {"bottle", "container", "spray", "tank"},
        "context_sensitive": True,
        "severity": "high"
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
    
    # Floor obstacles - objects that shouldn't be on the floor
    {"bag"},
    {"backpack"},
    {"suitcase"},
    {"box"},
    {"cord"},
    {"cable"},
    {"bottle"},
    
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

# Add cooldown tracking
HAZARD_COOLDOWN = 9  # seconds (3x the base cooldown of 3 seconds)
last_hazard_detection = {}  # Store timestamp of last detection for each hazard type
critical_alert_count = 0  # Track number of critical alerts

def is_hazard_in_cooldown(hazard_type):
    """Check if a hazard type is still in cooldown period"""
    global last_hazard_detection
    
    current_time = datetime.now()
    last_time = last_hazard_detection.get(hazard_type)
    
    if last_time is None:
        return False
        
    time_diff = current_time - last_time
    return time_diff.total_seconds() < HAZARD_COOLDOWN

def update_hazard_timestamp(hazard_type):
    """Update the last detection timestamp for a hazard type"""
    global last_hazard_detection, critical_alert_count
    
    last_hazard_detection[hazard_type] = datetime.now()
    
    # Increment critical alert count if severity is critical
    if hazard_type in HAZARD_CATEGORIES and HAZARD_CATEGORIES[hazard_type].get("severity") == "critical":
        critical_alert_count += 1
        
        # Check if we've reached the threshold
        if critical_alert_count >= 5:
            trigger_webhook()
            critical_alert_count = 0  # Reset counter after triggering

def calculate_box_overlap(box1, box2):
    """
    Calculate how much box1 overlaps with box2.
    Returns the percentage of box1 that is inside box2.
    """
    # Get coordinates
    x1_min, y1_min = box1["box"][0], box1["box"][1]
    x1_max, y1_max = box1["box"][2], box1["box"][3]
    x2_min, y2_min = box2["box"][0], box2["box"][1]
    x2_max, y2_max = box2["box"][2], box2["box"][3]
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    # Return percentage of box1 that overlaps with box2
    return intersection_area / box1_area if box1_area > 0 else 0.0

def is_object_held_safely(person_or_hand_box, object_box, overlap_threshold=0.3):
    """
    Determine if an object is being held safely by checking if it overlaps significantly
    with a person or hand bounding box
    """
    # If it's a hand detection, use the original distance check
    if person_or_hand_box["label"] == "hand":
        x1_hand = (person_or_hand_box["box"][0] + person_or_hand_box["box"][2]) / 2
        y1_hand = (person_or_hand_box["box"][1] + person_or_hand_box["box"][3]) / 2
        x2_obj = (object_box["box"][0] + object_box["box"][2]) / 2
        y2_obj = (object_box["box"][1] + object_box["box"][3]) / 2
        
        # Calculate distance between centers
        distance = np.sqrt((x1_hand - x2_obj)**2 + (y1_hand - y2_obj)**2)
        return distance <= 50  # Original safe distance for hands
    
    # For person detection, check box overlap
    overlap = calculate_box_overlap(object_box, person_or_hand_box)
    return overlap >= overlap_threshold

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
    Classify hazards based on detected objects and their relationships.
    Returns:
    - is_hazardous (bool): Whether a hazard is present
    - hazard_types (list): List of detected hazard types
    - severity (str): Overall severity level (critical, high, medium, low)
    """
    global critical_alert_count
    hazard_types = []
    max_severity = "low"
    severity_levels = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    has_person = "person" in labels
    
    # Check for floor obstacles first
    floor_objects = HAZARD_CATEGORIES["floor_obstacles"]["objects"]
    for label in labels:
        if label in floor_objects:
            if not is_hazard_in_cooldown("floor_obstacles"):
                hazard_types.append("floor obstacles")  # Using space instead of underscore
                update_hazard_timestamp("floor_obstacles")
                max_severity = "critical"  # Floor obstacles are always critical
                critical_alert_count += 1
                logger.info(f"Critical alert count (floor obstacle): {critical_alert_count}")
                if critical_alert_count >= 5:
                    if trigger_webhook():
                        critical_alert_count = 0
    
    # Check for immediate hazards that don't require human presence
    for category, rules in HAZARD_CATEGORIES.items():
        if rules.get("always_hazard", False):
            for obj in rules["objects"]:
                if obj in labels:
                    # Check cooldown before adding hazard
                    if not is_hazard_in_cooldown(category):
                        # Convert category name to readable format
                        readable_category = category.replace("_", " ")
                        hazard_types.append(readable_category)
                        update_hazard_timestamp(category)
                        if rules.get("severity") == "critical":
                            critical_alert_count += 1
                            logger.info(f"Critical alert count: {critical_alert_count}")
                            if critical_alert_count >= 5:
                                if trigger_webhook():
                                    critical_alert_count = 0
                        max_severity = max(max_severity, rules.get("severity", "low"), 
                                        key=lambda x: severity_levels.get(x, 0))
    
    # Check for unattended sharp objects - these are CRITICAL
    if not has_person:
        for obj in HAZARD_CATEGORIES["sharp_objects"]["objects"]:
            if obj in labels:
                if not is_hazard_in_cooldown("unattended_sharp_objects"):
                    hazard_types.append("unattended sharp object")
                    update_hazard_timestamp("unattended_sharp_objects")
                    max_severity = "critical"
                    critical_alert_count += 1
                    logger.info(f"Critical alert count (unattended sharp object): {critical_alert_count}")
                    if critical_alert_count >= 5:
                        if trigger_webhook():
                            critical_alert_count = 0
                    break  # Only need to trigger once for multiple sharp objects
    
    # Check for hazardous combinations
    detected_objects = set(labels)
    for combo in HAZARD_COMBINATIONS:
        if combo.issubset(detected_objects):
            hazard_category = None
            # Determine hazard category based on combination
            if "ladder" in combo or "scaffold" in combo:
                hazard_category = "fall_hazards"
            elif "knife" in combo or "scissors" in combo:
                hazard_category = "sharp_objects"
            elif "fire" in combo or "smoke" in combo:
                hazard_category = "fire_hazards"
            
            if hazard_category and not is_hazard_in_cooldown(hazard_category):
                hazard_types.append(hazard_category)
                update_hazard_timestamp(hazard_category)
                severity = HAZARD_CATEGORIES[hazard_category].get("severity", "low")
                if severity == "critical":
                    critical_alert_count += 1
                    logger.info(f"Critical alert count: {critical_alert_count}")
                    if critical_alert_count >= 5:
                        if trigger_webhook():
                            critical_alert_count = 0
                max_severity = max(max_severity, severity,
                                key=lambda x: severity_levels.get(x, 0))
    
    # Analyze spatial relationships
    spatial_hazards = analyze_spatial_relationships(boxes)
    for hazard in spatial_hazards:
        hazard_type = hazard["type"]
        if not is_hazard_in_cooldown(hazard_type):
            hazard_types.append(hazard_type)
            update_hazard_timestamp(hazard_type)
            if hazard["severity"] == "critical":
                critical_alert_count += 1
                logger.info(f"Critical alert count: {critical_alert_count}")
                if critical_alert_count >= 5:
                    if trigger_webhook():
                        critical_alert_count = 0
            max_severity = max(max_severity, hazard["severity"],
                            key=lambda x: severity_levels.get(x, 0))
    
    logger.info(f"Hazard classification: types={hazard_types}, severity={max_severity}, has_person={has_person}")
    return bool(hazard_types), hazard_types, max_severity

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

def trigger_webhook():
    """Trigger webhook after threshold of critical alerts"""
    try:
        webhook_url = "https://haraghav.app.n8n.cloud/webhook-test/e4f0165a-70a2-47de-9dec-00eccd6ba7b6"
        payload = {
            "event": "critical_hazard_threshold",
            "timestamp": datetime.now().isoformat(),
            "alert_count": critical_alert_count,
            "message": "Critical hazard threshold reached - immediate attention required"
        }
        logger.info(f"Attempting to trigger webhook with payload: {payload}")
        response = requests.post(webhook_url, json=payload, timeout=5)  # 5 second timeout
        
        if response.status_code == 200:
            logger.info("✓ Webhook triggered successfully")
            return True
        else:
            logger.error(f"✗ Webhook trigger failed with status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return False
    except requests.exceptions.Timeout:
        logger.error("✗ Webhook trigger timed out after 5 seconds")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Error triggering webhook: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error triggering webhook: {str(e)}")
        return False