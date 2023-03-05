import os
import re
import sys
import openai
import PyPDF2
import requests

from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm
from pprint import pprint

# Define a regular expression pattern to match the thought types
pattern = r"^(.*),\s*(PREDICTION|REFLECTION|STATEMENT|QUESTION)$"

models = {
    "semantic": "davinci:ft-personal:semantic-iris-davinci-3-2022-11-30-06-30-47",
    "davinci": "text-davinci-003",
    "thought_type": ""
}

# Usage
# python3 parse_claims.py http://example.com
# python3 parse_claims.py /path/to/my/pdf/purplepage.pdf

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the maximum length of each text chunk to send to OpenAI
chunk_length = 3000

# Get the input file path from the command line argument
input_file = sys.argv[1]

# Check if the input file is a URL or a PDF file
if input_file.startswith('http'):
    
    # set up the browser
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('disable-gpu')
    driver = webdriver.Chrome(options=options)

    # navigate to the page
    driver.get(input_file)

    # get the page source with JavaScript
    html_content = driver.page_source

    # close the browser
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    driver.quit()

else:
    # Input file is a PDF file
    if input_file.endswith('.pdf'):
        with open(input_file, 'rb') as f:
            try:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ''
                for i in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[i].extract_text()
            except PyPDF2.utils.PdfReadError:
                print("Failed to read PDF file.")
                sys.exit()
    else:
        print("Invalid input file format. Please provide a URL or a PDF file.")
        sys.exit()

# Clean the Text
text = text.replace('\n', ' ')

# Split the text into individual sentences
sentences = text.split('. ')

# Capitalize the first letter and make the remaining letters lowercase in each sentence
formatted_sentences = []
for sentence in sentences:
    formatted_sentence = sentence.capitalize()
    formatted_sentence = formatted_sentence[0] + formatted_sentence[1:].lower()
    formatted_sentences.append(formatted_sentence)

# Join the sentences back into a single string
text = '. '.join(formatted_sentences)

# Split the text into chunks of maximum length
text_chunks = [text[i:i+chunk_length] for i in range(0, len(text), chunk_length)]

# Send each text chunk to OpenAI for processing with progress bar

explanation = "The fourThought dialectic is about the tensing of the language and the certainty behind it. It has four thought types: predictions, statements, reflections and questions. Predictions are claims about the future, reflections are claims about the past, statements are claims about the present. Questions are not declarations of truth. They are the opposite, they query the world based on uncertainty.\n\nI want you to read this text and break it into predictions, reflections, statements and questions.\n\nPrint each thought like this THOUGHT_TEXT, THOUGHT_TYPE. \n\nThe world will continue on well past 2042, PREDICTION\nIn the early 1900's the Spanish flu became a pandemic, REFLECTION\nThe world is a vampire, STATEMENT\nThe future is bright and prosperous, PREDICTION\nThis will require new tools PREDICTION\nWhere did I leave my keys? QUESTION?\n\nThe text you are receiving is scraped from the internet. It may be noisy and contain non-standard capitalization. Strip white space and recase the claims to be readable by humans. Follow the format above in printing with one line between each discrete thought.\n\nYou will be reading text from the internet and parsing it into four thought types. Reflections are about the past. Statements are about now. Predictions are about the future. Predictions can't be verified until time passes. Memories can be verified by referencing a record. If a thought is a reflection we should be able to check the record to see if it matches. Statements are about now and can be verified through a democratic process of asking the community. Predictions can not be verified immediately. Time must pass for predictions to be verified."
claims = []
conversation = [{"role": "system", "content": "You are a kind and prophetic assistant that takes in noisy text scraped from the internet parses it into the FourThought Dialectic: Predictions, Statements, Reflections and Questions"}]
conversation.append({"role": "user", "content": explanation})

for i, text_chunk in tqdm(enumerate(text_chunks), total=len(text_chunks), unit='chunk'):

    # Prepend the text with the prompt string
    text_prompt = "Here is the current chunk of text \n" + text_chunk + "\n"
    conversation.append({"role": "user", "content": text_prompt})
    truncated_convo = []
    truncated_convo.append({"role": "user", "content": explanation})
    truncated_convo.append({"role": "assistant", "content": "Okay I will print one thought per line with the thought type at the end. Thoughts about the future will be labeled as PREDICTION and thoughts about the past will be labeled as REFLECTION. All other thoughts will be labeled STATEMENT or QUESTION"})
    truncated_convo += conversation[-4:]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=truncated_convo
    )

    # Extract the claims from the response and add them to the claims array
    for choice in response.choices:
        assistant_message = choice.message.content
        conversation.append({"role": "assistant", "content": assistant_message})
        claims += [c.strip() for c in assistant_message.split("\n") if c.strip()]

# Pretty-print the array of claims
for claim in claims:
    match = re.match(pattern, claim)
    if match:
        print(match.group(1))
        print("\n")