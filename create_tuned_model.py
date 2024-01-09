import google.generativeai as genai
import google.api_core
from dotenv import load_dotenv
import os
import asyncio
import time

load_dotenv()

# The api key for accessing the api. stored in .env
API_KEY = os.getenv("API_KEY")

# Set the environment variable for the service account key file.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


def get_base_model():
    """Retrieves the first available model with text tuning capabilities.

        Uses the GenAI library to fetch a list of available models and returns the
        first model that supports the "createTunedTextModel" method.

        Returns:
            Model: The first model with text tuning capabilities.
        """
    # Get a list of available models from GenAI
    available_models = genai.list_models()

    # Filter models to those supporting text tuning
    filtered_models = [
        m for m in available_models
        if "createTunedTextModel" in m.supported_generation_methods
    ]

    # Retrieve the first model that supports text tuning
    base_model = filtered_models[0]

    return base_model


def check_for_existing_tuned_models():
    """Prints a list of available base models and tuned models.

    Utilizes the GenAI library to list both:
      - Base models that can be used for tuning.
      - Tuned models that have already been created.

    This function is primarily for informative purposes, allowing the user
    to see which models are available for tuning and which tuned models already
    exist in their account.
    """

    # Retrieve and print a list of available base models
    print('Available base models:', [m.name for m in genai.list_models()])

    # Retrieve and print a list of existing tuned models
    print('My tuned models:', [m.name for m in genai.list_tuned_models()])


def create_tuned_model(name, base_model, training_data):
    """Creates a tuned model using GenAI's tuning functionality.

    Args:
        name (str): The desired name for the tuned model.
        base_model (Model): The base model to use for tuning.
        training_data (str): The path to the training data file.

    Raises:
        ValueError: If the base model does not support text tuning.
    """

    # Ensure the base model supports text tuning
    if "createTunedTextModel" not in base_model.supported_generation_methods:
        raise ValueError("Base model does not support text tuning.")

    # Initiate the model tuning process through GenAI
    operation = genai.create_tuned_model(
        source_model=base_model.name,  # Specify the base model to use
        training_data=training_data,  # Provide the path to the training data
        id=name,  # Set the desired name for the tuned model
        epoch_count=100,  # Specify the number of training epochs
        batch_size=4,  # Set the batch size for training
        learning_rate=0.001,  # Set the learning rate for the optimizer
    )

    # Monitor the tuning progress using a wait_bar
    for status in operation.wait_bar():
        print(status)  # Print status updates during model tuning
        time.sleep(30)  # Pause for 30 seconds between status checks


def delete_tuned_model(delete_model_name):
    genai.delete_tuned_model(delete_model_name)


async def main():
    """Main function to initiate model tuning with GenAI.

    This function performs the following steps:
    1. Prints a greeting message to the console.
    2. Retrieves a suitable base model for tuning.
    3. Prints the name of the selected base model.
    4. Prints lists of available base models and existing tuned models.
    5. Creates a tuned model using the specified name, base model, and training data.
    """

    print("Create a tuned model: ")  # Greeting message

    model_name = "candy-region-model-1"
    training_data = "tuning_data\candy_location.json"

    base_model = get_base_model()  # Retrieve a model for tuning
    print(f'base model name: {base_model.name}')  # Print the name of the retrieved model

    try:
        create_tuned_model(
            model_name,  # Desired name for the tuned model
            base_model,  # Base model to use for tuning
            training_data  # Path to the training data
        )
    except google.api_core.exceptions.InternalServerError as e:
        print(f"Google broke something: {e} ")
    except Exception as e:
        # delete the model because it probably doesn't work
        delete_tuned_model(f'tunedModels/{model_name}')
        print(f"Exception: {e} ")

# For testing tuned models.
if __name__ == "__main__":

    # deleting junk models.
    # delete_tuned_model('tunedModels/candy-region-model-2-2')
    # check_for_existing_tuned_models()
    asyncio.run(main())
