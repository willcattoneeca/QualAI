# GPT Sentience Dataset

This repository contains a dataset designed to fine-tune GPT-3.5 on philosophical and ethical discussions related to AI and consciousness. 

## Usage
### 1. Clone the repository.
### 2. Set Up a Virtual Environment
    1. Install venv (if not already installed):
    ```
    sudo apt update
    sudo apt install python3-venv
    ```
    2. Create a Virtual Environment:
    ```
    python3 -m venv venv
    ```
    3. Activate the Virtual Environment:
    ```
    source venv/bin/activate
    ```
### 3. Install Dependencies
    1. Ensure pip is up to date:
    ```
    python3 -m pip install --upgrade pip
    ```
    2. Install Required Packages:
    ```
    pip install -r requirements.txt
    ```
    3. Verify Installation: Check that FastAPI, Uvicorn, and OpenAI are installed by running:
    ```
    pip list
    ```
### 4. Project Structure
* `data/sentience_examples.jsonl`: Dataset of multi-turn conversations exploring AI's perspective on sentience and consciousness.
* `app.py`: Backend server for interacting with the fine-tuned model using FastAPI.
* `static/index.html`: Frontend chat interface for seamless interaction.
* `requirements.txt`: File listing all Python dependencies for the project.
### 5. Fine-Tuning the Model
Prepare the dataset:
```
openai tools fine_tunes.prepare_data -f data/sentience_examples.jsonl
```
Upload the dataset:
```
openai file create --file data/sentience_examples.jsonl --purpose fine-tune
```
Start the fine-tuning process:
```
openai fine_tunes.create -t <file-id> -m "gpt-3.5-turbo"
```
Use the fine-tuned model in the backend: Update the model ID in app.py.

### 6. Use the dataset to fine-tune GPT-3.5 using the OpenAI API.

## Fine-Tuning with OpenAI API
- Install the OpenAI CLI: `pip install openai`.
- Prepare the dataset: Ensure it's in JSONL format.
- Upload the dataset: `openai file create --file data/sentience_examples.jsonl --purpose fine-tune`.
- Start fine-tuning: Use the uploaded file ID to fine-tune the model.
