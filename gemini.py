from google.ai import generativelanguage_v1beta
import asyncio
from pathlib import Path
import google.api_core
import gemini_util


def read_prompt():
    with open("prompt.txt", "r") as file:
        return file.read()


async def sample_generate_text_image_content(text, image):
    """
        This function sends a text and image reqeust to gemini.

        :param text: The text prompt from the user.
        :param image: The image from the user as a byte array
        :return: The response from gemini
        """

    # Create a client
    client = generativelanguage_v1beta.GenerativeServiceAsyncClient()

    request = generativelanguage_v1beta.GenerateContentRequest(
        model=gemini_util.IMAGE_MODEL_NAME,
        contents=[gemini_util.build_content("user", image, f'{read_prompt} {text}')],
        generation_config=gemini_util.generation_config,
        safety_settings=gemini_util.safety_settings
    )

    # Make the request
    response = await client.generate_content(request=request)

    # Handle the response
    return response.candidates[0].content.parts[0].text


async def sample_generate_text_content(text_list):
    """
        This function sends a text reqeust to gemini.

        :param text_list:
        :param text: The text prompt from the user.
        :return: The response from gemini
        """
    # Create a client
    client = generativelanguage_v1beta.GenerativeServiceAsyncClient()

    contents = []
    for text in text_list:
        role = "user"
        if text.find("Gemini: ") != -1:
            text = text.replace("Gemini: ", "")
            role = "model"
        contents.append(gemini_util.build_content_text(role, text))

    # for obj in contents:
    #    print({"parts": obj.parts, "role": obj.role})

    try:
        # Initialize request argument(s)
        request = generativelanguage_v1beta.GenerateContentRequest(
            model=gemini_util.TEXT_MODEL_NAME,
            contents=contents,
            generation_config=gemini_util.generation_config,
            safety_settings=gemini_util.safety_settings
        )

        # Make the request
        response = await client.generate_content(request=request)

        # Handle the response
        return response.candidates[0].content.parts[0].text
    except google.api_core.exceptions.FailedPrecondition as e:
        # print('Failed precondition error:', e)
        return f'Error: {e}  Tip: use a VPN.'
    except Exception as e:
        # print('Unknown error:', e)
        return f'Error: {e}'


async def main():
    # Gemini provides a multimodal model (gemini-pro-vision) that accepts both text and images and inputs. The
    # GenerativeModel.generate_content API is designed handle multi-media prompts and returns a text output.

    image_bites = Path("image.jpg").read_bytes()
    response = await sample_generate_text_image_content('What do you see?', [image_bites])
    print(f'Text Image response: {response}')

    # for testing text based calls.
    response = await sample_generate_text_content(["Who is james T. Kirk?"])
    print(f'Text response: {response}')


async def main_text(text):
    response = await sample_generate_text_content(text)
    return f'Gemini: {response}'


async def main_chat():
    # For testing images chat
    lines = []
    while True:
        user_input = input(": ")  # Prompt the user
        lines.append(user_input)
        data = await main_text(lines)
        lines.append(data)
        print(data)


# For testing text based chat.
if __name__ == "__main__":
    # test text based and image.
    # asyncio.run(main())

    # test chat
    asyncio.run(main_chat())
