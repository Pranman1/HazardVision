import google.generativeai as genai
import requests
import base64
import json
import os
from datetime import datetime

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
        # Prepare the system prompt
        system_prompt = """You are a workplace safety expert AI. Analyze the provided image and hazard information to:
1. Describe the exact safety violation observed
2. Explain why this is dangerous
3. Provide a clear, concise recommendation for immediate correction
Keep the total response under 100 words and focus on actionable insights.
Format: "Safety Alert: [brief description]. This is dangerous because [reason]. Recommended Action: [action]."
"""
        
        # Load and encode image
        image_data = encode_image_base64(image_path)
        
        # Create the prompt with context
        prompt = f"Analyze this workplace safety situation. Detected hazards: {', '.join(hazard_types)}. Severity: {severity}."
        
        # Get LLM response
        response = model.generate_content([system_prompt, prompt, image_data])
        
        return response.text
        
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return "Critical safety hazard detected. Please check the area immediately."

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
            audio_path = f"frontend/static/alerts/alert_{timestamp}.mp3"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            with open(audio_path, "wb") as f:
                f.write(response.content)
            
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