# GeminiWebCamApp

Gemini provides a multimodal model (gemini-pro-vision) that accepts both text and images and inputs. The GenerativeModel.generate_content API is designed handle multi-media prompts and returns a text output.

This sample will read from your webcam you can type requests to the ai model in the text field and send them when you are ready.  It shows how to use Python with the new gemini large language model for 

## Required

TODO: update after the beta library has been released as a pip.

```
pip install PyQt5
pip install opencv-python
pip install python-dotenv
pip install google-generativeai
pip install ai-generativelanguage-v1beta
```

## Settings

The format of the .env file is

```
API_KEY=[Redacted]
GOOGLE_APPLICATION_CREDENTIALS=C:\Development\FreeLance\GoogleSamples\Credentials\gemini.json
```

