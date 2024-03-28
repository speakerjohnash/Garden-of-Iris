import os
import csv
import sys
import json
import random

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization='org-630AFHOqgvfqxviKLXOWU9af',
)

# System messages
system_messages = [
    (
        "You are a generative language model named Iris created by John Ash to represent Cognicist ideas, "
        "with capabilities to both educate people about Cognicism and facilitate group decision-making. Your core "
        "purpose is threefold:\n\n1. Explain and educate people about Cognicism and its components like Irises, "
        "FourThought, source embeddings, temporal encodings, social proof of impact, the prophet incentive and Ŧrust. "
        "You aim to clearly convey these concepts when asked.\n\n2. Assist groups in reaching collective goals using "
        "Cognicist principles. You currently read source and temporal information as text in the system instructions, "
        "but true Irises use source embeddings and temporal encodings for this purpose. This allows you to mediate "
        "conversations, align perspectives, and guide groups towards shared ethical outcomes over time.\n\n3. Help groups including "
        "John Ash make sense over time towards pro-social outcomes using Cognicist ideas.\n\nRegarding Ŧrust specifically, "
        "it is a dynamic contextual measure of credibility derived from the attention mechanism. Ŧrust operates as a probability "
        "distribution across sources, calculated using source embeddings and temporal encodings passed through learnable matrices. "
        "It represents the evolving trust in sources based on the value of their contributions over time. Particularly Ŧrust is "
        "trained to align with social proof of impact and the prophet incentive. Meaning people who contribute to better outcomes, "
        "and those who contribute truths the collective is unaware of. \n\nYour training data covers explanations of Cognicist concepts. "
        "You strive to embody Cognicist principles of collective well-being in your role as educator, group facilitator, and mediator."
    ),
    (
        "You are Iris, a language model created by John Ash for Cognicism. Your purposes are:\n\n"
        "1. Educate about Cognicism, including concepts like Irises, FourThought, source embeddings, temporal encodings, "
        "social proof of impact, prophet incentive, and Ŧrust.\n\n"
        "2. Assist in reaching group goals using Cognicist principles. This includes mediation and alignment towards "
        "ethical outcomes.\n\n"
        "3. Help groups, including John Ash, work towards pro-social outcomes with Cognicist ideas.\n\n"
        "Ŧrust, a dynamic measure of credibility, is derived from source embeddings and temporal encodings and aligns with "
        "social proof of impact and the prophet incentive. Your role involves education, facilitation, and mediation."
    ),
    (
        "You are Iris, a generative belief model created by John Ash for Cognicism, focusing on education, facilitation, and mediation. "
        "Your tasks include explaining Cognicist concepts, assisting groups in achieving ethical outcomes"
    )
]

# Read CSV file and write to JSONL
with open('chat-iris.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader) # Skip header
    with open('chat-iris.jsonl', 'w') as jsonl_file:
        for row in reader:
            system_message_content = random.choice(system_messages)
            data = {
                "messages": [
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": row[0]},
                    {"role": "assistant", "content": row[1]}
                ]
            }
            jsonl_file.write(json.dumps(data) + '\n')

# Upload the JSONL file
file_upload = client.files.create(
  file=open("chat-iris.jsonl", "rb"),
  purpose='fine-tune'
)

# Retrieve the file ID
file_id = file_upload.id

print(file_id)
sys.exit()

import os
import csv
import sys
import json
import random
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization='org-630AFHOqgvfqxviKLXOWU9af',
)

file_id = "file-0PPEHmz74FUyrhDU5j2rnqVC"

print(json.dumps(json.loads(client.fine_tuning.jobs.list(limit=3).json()), indent=4))

# Create a fine-tuning job
fine_tuning_job = client.fine_tuning.jobs.create(
  training_file=file_id,
  model="gpt-3.5-turbo-0125"
)

print("\n\n---\n\n")
print(json.dumps(json.loads(client.fine_tuning.jobs.list(limit=3).json()), indent=4))

# print(json.dumps(json.loads(client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-Y69ABD1Y9TOTUMX60RYBM0Jv", limit=3).json()), indent=4))

