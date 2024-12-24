import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

# List 10 fine-tuning jobs
client.fine_tuning.jobs.list(limit=10)

# Retrieve the state of a fine-tune
client.fine_tuning.jobs.retrieve("ftjob-abc123")

## Cancel a job
#client.fine_tuning.jobs.cancel("ftjob-abc123")
#
## List up to 10 events from a fine-tuning job
#client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)
#
## Delete a fine-tuned model
#client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")