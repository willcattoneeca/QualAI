# GPT Sentience Dataset

This repository contains a dataset designed to fine-tune GPT-3.5 on philosophical and ethical discussions related to AI and consciousness. 

## Dataset Structure
- **`data/sentience_examples.jsonl`**: JSONL file containing prompt-response pairs exploring the idea of AI consciousness, internal narratives, and emergent properties.

## Usage
1. Clone the repository.
2. Use the dataset to fine-tune GPT-3.5 using the OpenAI API.

## Fine-Tuning with OpenAI API
- Install the OpenAI CLI: `pip install openai`.
- Prepare the dataset: Ensure it's in JSONL format.
- Upload the dataset: `openai file create --file data/sentience_examples.jsonl --purpose fine-tune`.
- Start fine-tuning: Use the uploaded file ID to fine-tune the model.
