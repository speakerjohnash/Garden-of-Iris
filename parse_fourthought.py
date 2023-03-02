import os
import sys
import openai
import PyPDF2
import requests

from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm
from pprint import pprint

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
claims = []

for i, text_chunk in tqdm(enumerate(text_chunks), total=len(text_chunks), unit='chunk'):
    # Prepend the text with the prompt string
    text_prompt = "The text you are receiving is scraped from the internet using beautiful soup and PyPDF2. That means it may be quite noisy and contain non-standard capitalization. We are working on a protocol for making sense of discrete claims of truth. These are simple sentences that are full thoughts and make some discrete claim of truth. Can you parse the following text into a list of well formatted claims in simple english? Strip white space and recase the claims to be readable by humans. Print them out in a list with no numbers with a space between each line" + text_chunk

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": text_prompt}]
    )

    # Extract the claims from the response and add them to the claims array
    for choice in response.choices:
        assistant_message = choice.message.content
        claims += [c.strip() for c in assistant_message.split("\n") if c.strip()]

# Pretty-print the array of claims
for claim in claims:
    print(claim)
    print("\n")