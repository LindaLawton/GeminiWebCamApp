import google.generativeai as genai
from google.ai import generativelanguage_v1beta
from dotenv import load_dotenv
import os
import requests
import asyncio
from pathlib import Path
import json

load_dotenv()

# The api key for accessing the api. stored in .env
API_KEY = os.getenv("API_KEY")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME")

genai.configure(api_key=API_KEY)
# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set up the model
generation_config = {
    'temperature': 0.9,
    'top_p': 1,
    'top_k': 40,
    'max_output_tokens': 2048,
    'stop_sequences': [],
}

safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                   {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                   {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                   {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]


model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


def create_image_part(image):
    image_blob = generativelanguage_v1beta.Blob(mime_type="image/jpeg", data=image)
    return generativelanguage_v1beta.Part(inline_data=image_blob)


def create_text_part(text):
    return generativelanguage_v1beta.Part(text=text)


def build_content(role, image_list, text):
    parts = []
    for image in image_list:
        if len(parts) >= 16:
            break  # Exit the loop after processing 16 images
        parts.append(create_image_part(image))

    parts.append(create_text_part(text))
    return generativelanguage_v1beta.Content(parts=parts, role=role)


def build_content_text(role, text):
    part = create_text_part(text)
    return generativelanguage_v1beta.Content(parts=[part], role=role)
