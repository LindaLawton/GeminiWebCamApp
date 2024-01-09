from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
import os

load_dotenv()

# The api key for accessing the api. stored in .env
genai.configure(api_key=os.getenv("API_KEY"))

# name of the ai model used in this call.
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")

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

model = genai.GenerativeModel(model_name=CHAT_MODEL_NAME,
                              generation_config=generation_config,
                              safety_settings=safety_settings)


def build_conversation_turn(role, text):
    return {"role": role, "parts": [text]}


def get_conversation(conversation):
    return model.start_chat(history=conversation)


async def main():
    try:
        # storing conversation data.
        conversation_history = []

        print("Test application for sending chat messages to Gemini ai.  To exit type Exit")
        print("To exit type Exit")

        while True:
            # Prompt the user to type something
            user_input = input(": ")

            # get conversation history
            convo = get_conversation(conversation_history)

            # send message with conversation history
            convo.send_message(user_input)

            if user_input.lower().strip() == "exit":
                print("Application shutting down.")
                break

            # This is for debugging the conversation history.  (Linda)
            if user_input.lower().strip() == "show history":
                print(conversation_history)
            else:
                # store user message as history
                conversation_history.append(build_conversation_turn("user", user_input))

                # store model response
                conversation_history.append(build_conversation_turn("model", convo.last.text))
                print(convo.last.text)

    except KeyboardInterrupt:
        print("Application shutting down. ")


# For testing text based chat.
if __name__ == "__main__":

    # NOTE this is a work in progress its kicking out a 500 error from googe

    # test text based and image.
    # asyncio.run(main())

    # test chat
    asyncio.run(main())
