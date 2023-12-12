import google.generativeai as genai
from google.ai import generativelanguage_v1beta
from dotenv import load_dotenv
import os
import requests
import asyncio
from pathlib import Path

load_dotenv()

# The api key for accessing the api. stored in .env
api_key = os.getenv("API_KEY")
path_to_service_account_key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
genai.configure(api_key=api_key)

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_service_account_key_file

# Set up the model
generation_config = {
    'temperature': 0.9,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048,
    'stop_sequences': [],
}

safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                   {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                   {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                   {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]


async def sample_generate_text_image_content(text, image):
    """
        This function sends a text and image reqeust to gemini.

        :param text: The text prompt from the user.
        :param image: The image from the user as a byte array
        :return: The response from gemini
        """

    # Create a client
    client = generativelanguage_v1beta.GenerativeServiceAsyncClient()

    image_blob = generativelanguage_v1beta.Blob(mime_type="image/jpeg", data=image)
    text_part = generativelanguage_v1beta.Part(text=text)
    image_part = generativelanguage_v1beta.Part(inline_data=image_blob)
    contents = generativelanguage_v1beta.Content(parts=[image_part, text_part], role="user")

    # Initialize request argument(s)
    request = generativelanguage_v1beta.GenerateContentRequest(
        model="models/gemini-pro-vision",
        contents=[contents],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Make the request
    response = await client.generate_content(request=request)

    # Handle the response
    return response.candidates[0].content.parts[0].text


async def sample_generate_text_content(text):
    """
        This function sends a text reqeust to gemini.

        :param text: The text prompt from the user.
        :return: The response from gemini
        """
    # Create a client
    client = generativelanguage_v1beta.GenerativeServiceAsyncClient()
    text_part = generativelanguage_v1beta.Part(text=text)
    contents = generativelanguage_v1beta.Content(parts=[text_part], role="user")

    # Initialize request argument(s)
    request = generativelanguage_v1beta.GenerateContentRequest(
        model="models/gemini-pro",
        contents=[contents],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Make the request
    response = await client.generate_content(request=request)

    # Handle the response
    return response.candidates[0].content.parts[0].text


async def main():
    # Gemini provides a multimodal model (gemini-pro-vision) that accepts both text and images and inputs. The
    # GenerativeModel.generate_content API is designed handle multi-media prompts and returns a text output.

    # downloading an image to test with
    if not os.path.exists("image.jpg"):
        image_url = "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg"
        response = requests.get(image_url)
        if response.status_code == 200:
            print("Image downloaded successfully")

            with open("image.jpg", "wb") as f:
                f.write(response.content)
        else:
            print("Error downloading image:", response.status_code)

    image_bites = Path("image.jpg").read_bytes()
    response = await sample_generate_text_image_content('What do you see?', image_bites)
    print(f'Text Image response: {response}')

    response = await sample_generate_text_content("Who is ames T. Kirk?")
    print(f'Text response: {response}')


# Just for testing

if __name__ == "__main__":
    asyncio.run(main())
