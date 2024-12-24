import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
model_to_train = os.getenv("MODEL_TO_TRAIN")
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key and model are set
if not api_key:
    raise ValueError("API key not found. Ensure OPENAI_API_KEY is set in your .env file.")
if not model_to_train:
    raise ValueError("Model to train not specified. Ensure MODEL_TO_TRAIN is set in your .env file.")

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Create a fine-tuning job
try:
    uploaded_file = client.files.create(
        file=open(os.path.join("data", "sentience_examples.jsonl"), "rb"),
        purpose="fine-tune"
    )
    print(f"File uploaded successfully. File Details: {uploaded_file}")

    job = client.fine_tuning.jobs.create(
        training_file=uploaded_file.id,
        model=model_to_train,
        method={
            "type": "supervised"
        },
    )
    print(f"Fine-tuning job created successfully: {job}")
except Exception as e:
    print(f"An error occurred while creating the fine-tuning job: {e}")