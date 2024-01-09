# GeminiWebCamApp

Gemini provides a multimodal model (gemini-pro-vision) that accepts both text and images and inputs. The GenerativeModel.generate_content API is designed handle multi-media prompts and returns a text output.

This sample will read from your webcam you can type requests to the ai model in the text field and send them when you are ready.  It shows how to use Python with the new gemini large language model for 


# to run this.

## for the main ui

    python app.py

## to just pay with gemini text based you can run

    python gemini.py

## to just run chat

    python gemini_chat.py

## WIP tuning models

Its broken on googles end

    python create_tuned_model.py

## Required

TODO: update after the beta library has been released as a pip.

```
pip install PyQt5
pip install opencv-python
pip install python-dotenv
pip install google-generativeai
pip install google-ai-generativelanguage
```

## Settings

The format of the .env file is

GOOGLE_APPLICATION_CREDENTIALS is path to the service account key file only needed for tuning

```
API_KEY=[readacted]
TEXT_MODEL_NAME=models/gemini-pro
IMAGE_MODEL_NAME=models/gemini-pro-vision
CHAT_MODEL_NAME=gemini-pro
GOOGLE_APPLICATION_CREDENTIALS=C:\Development\FreeLance\GoogleSamples\Credentials\gemini-407220-ea106e0d696d.json
```

## error 

If you see the following error try running a vpn.

>400 User location is not supported for the API use.


Text based chat conversation calls must go as user, model, user, model.  This error occurs if you dont send it in that format

> # 400 Please ensure that multiturn requests ends with a user role or a function response. 

 
