import google.generativeai as genai
import requests
import base64
import json
import os
from datetime import datetime
from PIL import Image

# API Configuration
GEMINI_API_KEY = "AIzaSyAl6X4cZYC2GE4KRTyOc__3k-ujX4ayEno"
ELEVEN_LABS_API_KEY = "sk_0cd0f86badc873f3524f5f349598f0bdca2291223d13754c"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

def encode_image_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_hazard_with_llm(image_path, hazard_types, severity):
    """Use Gemini to analyze the hazard situation"""
    try:
        # Prepare the system prompt with more specific context
        system_prompt = """You are a workplace safety expert AI. Your task is to analyze workplace safety situations and provide specific, actionable feedback. Focus on:
1. The exact hazard type(s) detected
2. The specific dangers they present
3. Clear, immediate actions needed to resolve the situation

Keep responses concise and actionable, under 100 words.
Format: "Safety Alert: [specific hazard]. This is dangerous because [specific reason]. Recommended Action: [specific action]."

Important: Never use underscores in your response. Use spaces instead.
"""
        
        # Load image using PIL
        image = Image.open(image_path)
        
        # Create a more detailed prompt with specific hazard context
        hazard_descriptions = {
            "fall_hazards": "potential falls from height",
            "floor obstacles": "items blocking walkways or creating trip hazards",
            "sharp_objects": "dangerous sharp tools or objects",
            "fire_hazards": "potential fire sources",
            "electrical_hazards": "unsafe electrical conditions",
            "chemical_hazards": "hazardous chemicals",
            "unattended sharp object": "sharp objects left unattended",
            "unsafe sharp object handling": "improper handling of sharp tools",
            "electrical hazard": "unsafe electrical situation",
            "bag": "bag left on floor creating obstacle",
            "backpack": "backpack blocking walkway",
            "box": "box creating trip hazard",
            "cord": "cord creating trip hazard",
            "cable": "cable creating trip hazard",
            "bottle": "bottle left on floor"
        }
        
        # Build detailed hazard description
        hazard_details = []
        for hazard in hazard_types:
            # Convert any underscores to spaces first
            hazard = hazard.replace("_", " ")
            if hazard in hazard_descriptions:
                hazard_details.append(hazard_descriptions[hazard])
            else:
                hazard_details.append(hazard)
        
        prompt = f"""Analyze this workplace safety situation.
Detected hazards: {', '.join(hazard_details)}.
Severity level: {severity.upper()}.
Provide specific details about the visible hazards and clear actions needed.
Remember: Use spaces instead of underscores in your response."""
        
        # Get LLM response
        response = model.generate_content([system_prompt, prompt, image])
        
        if response.text:
            # Ensure no underscores in response
            return response.text.replace("_", " ")
        else:
            return f"Safety Alert: {', '.join(hazard_details)} detected. Severity: {severity}. Immediate inspection required."
        
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        # Ensure no underscores in error message
        hazard_list = [h.replace("_", " ") for h in hazard_types]
        return f"Safety Alert: {', '.join(hazard_list)} detected. Severity: {severity}. Please check the area immediately."

def generate_speech(text):
    """Generate speech using ElevenLabs API"""
    try:
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"  # Using Rachel voice
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_LABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Save the audio file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Update path to use the correct frontend directory
            audio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "static", "alerts", f"alert_{timestamp}.mp3")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            with open(audio_path, "wb") as f:
                f.write(response.content)
            
            # Return the correct URL path for the frontend
            return f"/static/alerts/alert_{timestamp}.mp3"
        else:
            print(f"Error generating speech: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error in speech generation: {str(e)}")
        return None

def process_critical_hazard(image_path, hazard_types, severity):
    """Process a critical hazard with LLM analysis and speech synthesis"""
    try:
        # Get LLM analysis
        analysis = analyze_hazard_with_llm(image_path, hazard_types, severity)
        
        # Generate speech
        audio_path = generate_speech(analysis)
        
        return {
            "analysis": analysis,
            "audio_path": audio_path
        }
    except Exception as e:
        print(f"Error processing critical hazard: {str(e)}")
        return None 