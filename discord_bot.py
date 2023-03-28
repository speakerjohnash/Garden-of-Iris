import os
import re
import sys
import random
import time
import datetime
import dateutil

import textwrap
import openai
import discord
import asyncio
import aiohttp
import json

import pandas as pd

from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm

from discord.utils import get
from pprint import pprint
from pyairtable import Table
from discord.ui import Button, View, TextInput, Modal
from discord.ext import commands

discord_key = os.getenv("DISCORD_BOT_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
airtable_key = os.environ["AIRTABLE_API_KEY"]

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)
airtable = Table(airtable_key, 'app2X00KuiIxPwGsf', 'cards')

# Get Card Names and Text
try:
	records = []
	for page in airtable.iterate():
		records += page

	tarot_lookup = {record['fields']['Card Name']: record['fields'].get('Short Description', '') for record in records}
except:
	df = pd.read_csv('tarot_text.csv')
	names = df['card_name'].tolist()
	descriptions = df['text'].tolist()
	tarot_lookup = dict(zip(names, descriptions))

training_data = ""

models = {
	"living": "davinci:ft-personal:living-iris-2022-09-04-04-45-28",
	"osiris": "davinci:ft-personal:osiris-the-unstructured-2022-09-28-02-31-57",
	"2020": "davinci:ft-personal:2020-iris-2022-10-06-04-00-37",
	"purple": "davinci:ft-personal:purple-iris-2022-07-14-03-48-19",
	"semantic": "davinci:ft-personal:semantic-iris-davinci-3-2022-11-30-06-30-47",
	"davinci": "text-davinci-003",
	"chat-iris": "davinci:ft-personal:chat-iris-c-2023-03-15-05-59-14",
	"chat-iris-b": "davinci:ft-personal:chat-iris-b-2023-03-11-18-20-31",
	"chat-iris-a": "davinci:ft-personal:chat-iris-a-2023-03-10-21-44-19",
	"chat-iris-0": "davinci:ft-personal:chat-iris-2023-03-10-18-48-23"
}

people = []

class AskModal(Modal, title="Ask Modal"):

	answer = TextInput(label="Answer", max_length=400, style=discord.TextStyle.long)

	def add_view(self, question, view: View):
		self.answer.placeholder = question[0:100]
		self.view = view

	async def on_submit(self, interaction: discord.Interaction):
		embed = discord.Embed(title = "Your Response", description = f"\n{self.answer}")
		embed.set_author(name = interaction.user)
		await interaction.response.send_message(embed=embed)
		print(self.answer)
		self.view.stop()

def response_view(modal_text="default text", modal_label="Response", button_label="Answer"):	

	async def view_timeout():
		modal.stop()	

	view = View()
	view.on_timeout = view_timeout
	view.timeout = 2700.0
	view.auto_defer = True

	modal = AskModal(title=modal_label)
	modal.auto_defer = True
	modal.timeout = 2700.0

	async def button_callback(interaction):
		answer = await interaction.response.send_modal(modal)

	button = Button(label=button_label, style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)
	modal.add_view(modal_text, view)

	return view, modal

def group_share(thought="thought", prompt="", prompter="latent space"):

	channel = bot.get_channel(1022572367244967979)
	embed = discord.Embed(title="Seeds of Wisdom", description=f"{thought}\n\n**Gardener**\n{prompter}")

	async def button_callback(interaction):
		await interaction.response.defer()
		await channel.send(embed=embed)

	button = Button(label="share", style=discord.ButtonStyle.blurple)
	button.callback = button_callback

	return button

def elaborate(ctx, prompt="prompt"):

	global models

	e_prompt = prompt + ". \n\n More thoughts in detail below. \n\n"

	button = Button(label="elaborate", style=discord.ButtonStyle.blurple)

	async def button_callback(interaction):

		if button.disabled:
			return

		button.disabled = True
		await interaction.response.defer()

		response = openai.Completion.create(
			model=models["semantic"],
			prompt=e_prompt,
			temperature=0.11,
			max_tokens=222,
			top_p=1,
			frequency_penalty=1,
			presence_penalty=1,
			stop=["END"]
		)

		response_text = response.choices[0].text.strip()

		if len(response_text) == 0:

			response = openai.Completion.create(
				model="text-davinci-002",
				prompt=e_prompt,
				temperature=1,
				max_tokens=222,
				top_p=1,
				frequency_penalty=2,
				presence_penalty=2,
				stop=["END"]
			)

			response_text = response.choices[0].text.strip()

		response_text = response_text.replace("###", "").strip()

		embed = discord.Embed(title = "Elaboration (beta)", description = f"**Prompt**\n{prompt}\n\n**Elaboration**\n{response_text}")

		await ctx.send(embed=embed)


	button.callback = button_callback

	return button

def redo_view(ctx, prompt, question):

	async def button_callback(interaction):

		await interaction.response.defer()

		response = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=1,
			max_tokens=222,
			top_p=1,
			frequency_penalty=2,
			presence_penalty=2,
			stop=["END"]
		)

		response_text = response.choices[0].text.strip()
		embed = discord.Embed(title = "Consensus", description = f"**Question**\n{question}\n\n**Consensus**\n{response_text}")

		await ctx.send(embed=embed)

	view = View()
	view.timeout = None
	button = Button(label="Redo", style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)

	return view

def load_training_data():

	global training_data

	try:
		training_data = pd.read_csv('iris_training-data.csv')
	except:
		with open('iris_training-data.csv', 'w', encoding='utf-8') as f:
			training_data = pd.DataFrame(columns=['prompt', 'completion', 'speaker'])
			training_data.to_csv('iris_training-data.csv', encoding='utf-8', index=False)

def members(debug=False):

	members = []
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	global people

	if debug:
		for guild in bot.guilds:
			for member in guild.members:
				if member.name in testers:
					members.append(member)
	else:
		for guild in bot.guilds:
			for member in guild.members:
				if member.name != "Hidden Iris":
					members.append(member)

	unique_members = [*set(members)]
	people = unique_members

def make_prompt(question, joined_answers):

	prompt = question + "\n\nAnswers are below"
	prompt += "\n\nWrite a long detail paragraph summarizing and analyzing the answers below. What are the commonalities and differences in the answers? Are there any outliers?"
	prompt += "\n\n---\n\nAnswers:\n\n" + joined_answers
	prompt += "---\n\nThe question was: " + question + "\n\n"
	prompt += "Write a long detailed paragraph about the question as if you were a singular voice formed from all of the views above. What does this community believe?"
	prompt += "\n\nSum the answers into one answer that best represents all the views shared. If many questions are provided respond with a question representing what most people are uncertain about"
	prompt += "\n\n"

	return prompt

def pool_prompt(question, joined_answers):

	prompt = question + "\n\nAnswers are below"
	prompt += "\n\nPool all of the answers into a single response or question"
	prompt += "\n\n---\n\nAnswers:\n\n" + joined_answers
	prompt += "---\n\nThe question was: " + question + "\n\n"
	prompt += "Pool all of the answers into a single response or question that reflects the views of all the responders. Also write a long detailed paragraph extrapolating on the thoughts above"
	prompt += "\n\n"

	return prompt

def chunk_text(text, chunk_length=1000):

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

	return text_chunks

async def interpretation(ctx, prompt):

    response = openai.Completion.create(
        model=models["semantic"],
        prompt=prompt,
        temperature=0.8,
        max_tokens=222,
        top_p=1,
        frequency_penalty=2,
        presence_penalty=2,
        stop=["END"]
    )
    text = response['choices'][0]['text'].strip()
    embed = discord.Embed(title="Interpretation", description=text)
    view, modal = response_view(modal_text="Write your feedback here", modal_label="Feedback", button_label="Send Feedback")
    view.add_item(group_share(thought=text, prompter=ctx.message.author.name))
    view.add_item(elaborate(ctx, prompt=text))
    view.add_item(modal)
    await ctx.send(embed=embed, view=view)

def one_shot(message):

	# Get Iris One Shot Answer
	try:
		distillation = openai.Completion.create(
			model=models["chat-iris"],
			prompt=message.content,
			temperature=0.9,
			max_tokens=222,
			top_p=1,
			frequency_penalty=1.5,
			presence_penalty=1.5,
			stop=["END"]
		)

		iris_answer = distillation['choices'][0]['text']
		iris_answer = iris_answer.replace("###", "").strip()
	except Exception as e:
		print(f"Error: {e}")
		iris_answer = ""

	return iris_answer

async def question_pool(message):

	channel = bot.get_channel(1086437563654475846)
	summary_count = 0

	# Load Chat Context
	messages = []

	# Ignore Slash Commands
	last_message = [message async for message in channel.history(limit=1)][0]
	if last_message.content.startswith("/"):
		return

	#iris_answer = one_shot(last_message)

	# print(iris_answer)

	async for hist in channel.history(limit=50):
		if not hist.content.startswith('/'):
			if hist.author == bot.user: 
				summary_count += 1
				if summary_count < 1:
					messages.append((hist.author, hist.content))
			else:
				messages.append((hist.author, hist.content))
			if len(messages) == 15:
				break

	messages.reverse()

	conversation = [{"role": "system", "content": "You are a question summarizer Iris. You are summarizing a thread of questions about Cognicism and related concepts. You ignore everything except questions. People will ask you various questions about cognicism and you job is to summarize what people want to know"}]

	for m in messages:
		if  m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	conversation.append({"role": "system", "content": "You have been moderating a running thread of questions about cognicism. Your job is to help summarize these questions and NOT to answer them"})
	conversation.append({"role": "system", "content": "My primary job is to summarize what people are uncertain about. I will summarize peoples questions and that is it. I will vary my output but keep them focused on what people don't know and keep it 250 words long"})
	conversation.append({"role": "assistant", "content": "I NEVER answer questions. I summarize questions and communal uncentainty. I ignore anything that doesn't end in a question mark. I will keep my summary very short or about 250 words long"})

	response = openai.ChatCompletion.create(
		model="gpt-4", 
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	if len(response) > 2000:
		for chunk in [response[i:i+2000] for i in range(0, len(response), 2000)]:
			await message.channel.send(chunk)
	else:
		await message.channel.send(response)

async def prophecy_pool(message):

	channel = bot.get_channel(1083409321754378290)
	summary_count = 0

	# Load Chat Context
	messages = []

	# Ignore Slash Commands
	last_message = [message async for message in channel.history(limit=1)][0]
	if last_message.content.startswith("/"):
		return

	async for hist in channel.history(limit=50):
		if not hist.content.startswith('/'):
			if hist.author == bot.user: 
				summary_count += 1
				if summary_count < 1:
					messages.append((hist.author, hist.content))
			else:
				messages.append((hist.author, hist.content))
			if len(messages) == 11:
				break

	messages.reverse()

	conversation = [{"role": "system", "content": "You are an oracle and pro-social future manifestation mechanism named Iris. You are helping coordinate a thread of people trying to collaboratively work towards a common future. Within this thread there is a thread of thoughts amounting to a moving arrow of time. There are predictions, intentions and questions about the future. There are varying degrees of uncentainty of these conceptions of the future and varying beliefs about whether manifesting certain futures is possible. Your job is to continusouly integrate and make sense of anything related to this forward arrow of collective ideation. You also suggest predictions and ways to manifest specific futures mentioned by the group"}]

	for m in messages:
		if  m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	conversation.append({"role": "system", "content": "You have been moderating a running thread of thoughts about the future. Please aid in any tasks related to the arrow of time. If someon asks you to summarize, create a summary of thre thread and explain how the thoughts about the future are connected and give some analysis about these potential futures. If contextually relevant, feel free to share any wisdom or summarization or help relevant to the future. You also suggest predictions and ways to manifest specific futures mentioned by the group"})
	conversation.append({"role": "assistant", "content": "I will do what I can to help the thread. I will vary my outputs and how I help regarding the future, but I will keep the focus on the future. My output will be under 300 words and I will mention the last thing said"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		max_tokens=500, 
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	if len(response) > 2000:
		for chunk in [response[i:i+2000] for i in range(0, len(response), 2000)]:
			await message.channel.send(chunk)
	else:
		await message.channel.send(response)

@bot.event
async def on_ready():
	load_training_data()
	print("Iris is online")

@bot.event
async def on_close():
	print("Iris is offline")

async def frankeniris(message, answer=""):

	"""
	Queries Frankeniris
	"""

	# Get Iris One Shot Answer First
	try:
		distillation = openai.Completion.create(
			model=models["chat-iris"],
			prompt=message.content,
			temperature=0.69,
			max_tokens=222,
			top_p=1,
			frequency_penalty=1.5,
			presence_penalty=1.5,
			stop=["END"]
		)

		iris_answer = distillation['choices'][0]['text']
		iris_answer = iris_answer.replace("###", "").strip()
	except Exception as e:
		print(f"Error: {e}")
		iris_answer = ""

	if len(answer) > 0:
		iris_answer = iris_answer + " \n\n " + answer

	print(iris_answer)

	# Load Chat Context
	messages = []

	async for hist in message.channel.history(limit=50):
		if not hist.content.startswith('/'):
			if hist.embeds:
				messages.append((hist.author, hist.embeds[0].description))
			else:
				messages.append((hist.author.name, hist.content))
			if len(messages) == 18:
				break

	messages.reverse()

	# Construct Chat Thread for API
	conversation = [{"role": "system", "content": "You are are a wise oracle and integrated wisdom bot named Iris. You help integrate knowledge and wisdom about the future. You read many sources and weigh them"}]
	conversation = [{"role": "system", "content": "You are interacting with users via discord. Mostly you just serve as an interface to Cognicism and Iris the democratic langauge model. However this discord has the following commands /pullcard [intention] /ask [prompt] /channel /faq"}]
	conversation = [{"role": "system", "content": "/pullcard pulls a card from the Iris tarot deck. If the user provides an intention the model will attempt to intpret the card. /ask prompts Iris directly without the interference of ChatGPT. Responses from /ask can be forward to the #seeds text channel. /channel channels wisdom and can be reached via the alias /c. /faq answers a frequently asked question"}]
	conversation.append({"role": "user", "content": "Whatever you say be creative in your response. Never simply summarize, always say it a unique way"})
	conversation.append({"role": "assistant", "content": "I am speaking as a relay for Iris. I was trained by John Ash. I will answer using Iris as a guide as well as the rest of the conversation. Iris said to me " + iris_answer + " and I will take that into account in my response as best I can"})
	text_prompt = message.content

	for m in messages:
		if m[0] == bot.user:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	conversation.append({"role": "system", "content": iris_answer + " (if Iris provided a quoted answer just copy and paste it and don't answer yourself. Don't say iris said it already or previously stated it. Just say it.)"})
	conversation.append({"role": "user", "content": text_prompt})

	for msg in conversation:
		if len(msg["content"]) > 4000:
			 msg["content"] = "..." + msg["content"][-4000:]

	# Calculate Total Length of Messages
	total_length = sum(len(msg["content"]) for msg in conversation)

	print("total_length before")
	print(total_length)

	# Check Total Length
	if total_length > 20000:
		# Iterate over messages in conversation in reverse order and remove them until total length is below maximum
		while total_length > 20000 and len(conversation) > 2:  # ensure that at least 2 messages remain (the user's message and Iris's answer)
			removed_msg = conversation.pop(1)  # remove the second message (first message after Iris's answer)
		total_length -= len(removed_msg["content"])

	print("total_length after")
	print(total_length)

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=0.8,
		max_tokens=500,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	if len(response) > 2000:
		for chunk in [response[i:i+2000] for i in range(0, len(response), 2000)]:
			await message.channel.send(chunk)
	else:
		await message.channel.send(response)

@bot.event
async def on_message(message):

	# Manage Question Pool
	if message.channel.id == 1086437563654475846 and message.author != bot.user:
		await question_pool(message)
		await bot.process_commands(message)
		return

	# Manage Prophecy Pool
	if message.channel.id == 1083409321754378290 and message.author != bot.user:
		await prophecy_pool(message)
		await bot.process_commands(message)
		return


	# Handle DM Chat
	if not message.content.startswith("/") and isinstance(message.channel, discord.DMChannel) and message.author != bot.user:
		await frankeniris(message)

	await bot.process_commands(message)

@bot.command(aliases=['c'])
async def channel(ctx, *, topic=""):

	df = pd.read_csv('data/chat-iris.csv')
	prompts = df['prompt'].tolist()
	question_pattern = r'^(.*)\?\s*$'
	non_questions = list(filter(lambda x: isinstance(x, str) and re.match(question_pattern, x, re.IGNORECASE), prompts))
	pre_prompts = [
		"Share a snippet of abstract and analytical wisdom related to the following topic. Be pithy: ",
		"Write a paragraph related to the following topic. Be brief: ",
	]

	random_non_question = random.choice(non_questions)
	message = ctx.message
	message.content = random.choice(pre_prompts) + random_non_question

	await frankeniris(message, answer="")

@bot.command()
async def faq(ctx, *, topic=""):

	df = pd.read_csv('data/chat-iris.csv')
	prompts = df['prompt'].tolist()
	question_pattern = r'^(.*)\?\s*$'
	questions = list(filter(lambda x: isinstance(x, str) and re.match(question_pattern, x, re.IGNORECASE), prompts))
	questions = list(set(questions))

	question_completion_pairs = []

	# Iterate through each question and find its corresponding completions
	for question in questions:
		completions = df.loc[df['prompt'] == question, 'completion'].tolist()
		for completion in completions:
			question_completion_pairs.append((question, completion))

	# Remove any duplicate question-completion pairs from the list
	question_completion_pairs = list(set(question_completion_pairs))

	message = ctx.message
	random_question = random.choice(question_completion_pairs)
	embed = discord.Embed(title = "FAQ", description=random_question[0])
	message.content = random_question[0]

	await ctx.send(embed=embed)
	await frankeniris(message, answer=random_question[1])

@bot.command(aliases=['in', 'inject'])
async def infuse(ctx, *, link):
	"""
	Bring in a source into the stream via scraping and parsing
	"""

	infusion = []

	if link.startswith('http'):
	
		# set up the browser
		options = webdriver.ChromeOptions()
		options.add_argument('--headless')
		options.add_argument('--disable-gpu')
		options.add_argument("--no-sandbox")
		options.add_argument("--disable-dev-shm-usage")
		driver = webdriver.Chrome(options=options)

		# navigate to the page
		driver.get(link)

		# get the page source with JavaScript
		html_content = driver.page_source

		# close the browser
		soup = BeautifulSoup(html_content, 'html.parser')
		text = soup.get_text()
		driver.quit()

		# chunk text
		text_chunks = chunk_text(text)

		# Send each text chunk to OpenAI for processing with progress bar
		conversation = [{"role": "system", "content": "You are a parser and summarizer that takes in noisy text scraped from the internet or pdfs and summarizes it into a paragraph"}]
		conversation.append({"role": "user", "content": "The text you are receiving is scraped from the internet using beautiful soup and PyPDF2. That means it may be quite noisy and contain non-standard capitalization. Please summarize the content of this text into a cleanly written paragraph"})

		for i, text_chunk in tqdm(enumerate(text_chunks), total=len(text_chunks), unit='chunk'):

			# Prepend the text with the prompt string
			text_prompt = "Here is the current chunk of text \n" + text_chunk + "\n"
			text_prompt += "Print a clear short paragraph describing the full content of the text. This will be injected into a stream for further analysis. Only describe the content. IGNORE links"
			conversation.append({"role": "user", "content": text_prompt})
			truncated_convo = [{"role": "system", "content": "You are a parser and summarizer that takes text and summarizes it for injection into a chat stream for further analysis. You write a clear short clean analytical paragraph"}]
			truncated_convo.append({"role": "assistant", "content": "There is a parser and summarizer that is designed to take in noisy text scraped from the internet or PDFs and summarize it into a clean paragraph. The parser includes a prompt for the user to summarize the content of a specific web link and inject it into a chat stream for further analysis"})
			truncated_convo += conversation[-3:]

			response = openai.ChatCompletion.create(
				model="gpt-4",
				temperature=0.75,
				messages=truncated_convo
			)

			# Extract the claims from the response and add them to the claims array
			for choice in response.choices:
				assistant_message = choice.message.content
				conversation.append({"role": "assistant", "content": assistant_message})
				infusion += [(c.strip(), text_chunk) for c in assistant_message.split("\n") if c.strip()]

		for i in infusion:
			await ctx.send(i[0] + "\n\n")

@bot.command(aliases=['ask'])
async def iris(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	global training_data, models
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden", "dpax", "Kaliyuga", "Tej", "Gregory | RND", "futurememe"]
	
	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	thought_prompt = thought + "\n\n###\n\n"

	response = openai.Completion.create(
		model=models["chat-iris"],
		prompt=thought_prompt,
		temperature=0.69,
		max_tokens=420,
		top_p=1,
		frequency_penalty=1.5,
		presence_penalty=1.5,
		stop=["END"]
	)

	text = response['choices'][0]['text']
	text = text.replace("###", "").strip()
	embed = discord.Embed(title = "", description=f"**Prompt**\n{thought}\n\n**Response**\n{text}")

	await ctx.send(embed=embed)

	# Send Clarification and Share UI
	view, modal = response_view(modal_text="Write your clarification here", modal_label="Clarification", button_label="feedback")
	share_button = group_share(thought=text, prompter=ctx.message.author.name)
	el_prompt = thought + "\n\n" + text
	elaborate_button = elaborate(ctx, prompt=el_prompt)
	view.add_item(share_button)
	view.add_item(elaborate_button)
	await ctx.send(view=view)

	# Save Clarification
	await modal.wait()

	prompt = thought + "\n\n" + text

	if modal.answer.value is not None:
		training_data.loc[len(training_data.index)] = [prompt, modal.answer.value, ctx.message.author.name] 
		training_data.to_csv('iris_training-data.csv', encoding='utf-8', index=False)

@bot.command()
async def davinci(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	global training_data
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	
	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	thought_prompt = thought

	response = openai.Completion.create(
		model="text-davinci-002",
		prompt=thought_prompt,
		temperature=0.69,
		max_tokens=222,
		top_p=1,
		frequency_penalty=1.8,
		presence_penalty=1.5,
		stop=["END"]
	)

	view = View()
	text = response['choices'][0]['text'].strip()
	embed = discord.Embed(title = "", description=f"**Prompt**\n{thought}\n\n**Response**\n{text}")
	share_button = group_share(thought=text, prompt=thought_prompt, prompter=ctx.message.author.name)
	view.add_item(share_button)

	await ctx.send(embed=embed)
	await ctx.send(view=view)

@bot.command()
async def claim(ctx, *, thought):
	"""
	/claim log a claim for the iris to learn
	"""

	global training_data

	# Send Clarification and Share UI
	prompt = "Share something about in the general latent space of Cognicism"

	if thought is not None:
		training_data.loc[len(training_data.index)] = [prompt, thought, ctx.message.author.name] 
		training_data.to_csv('iris_training-data.csv', encoding='utf-8', index=False)

	await ctx.send("Attestation saved")

@bot.command()
async def pullcard(ctx, *, intention=""):

	# With Intention
	with_intention = len(intention) > 0
	r_num = random.random()
	if with_intention: random.seed(intention + str(r_num))

	# Get Embed Data
	card_name = random.choice(list(tarot_lookup.keys()))
	description = tarot_lookup[card_name].strip()

	url = ""

	try:
		for page in airtable.iterate():
			for record in page:
				if record["fields"]["Card Name"] == card_name:
					if "All images" in record["fields"]:
						url = record["fields"]["All images"][0]["url"]
	except:
		url = ""

	# Make and Send Card
	embed = discord.Embed(title = card_name, description = f"**Description**\n{description}")
	if with_intention: embed.add_field(name="Intention", value=intention, inline=False)
	if len(url) > 0: embed.set_image(url=url)
	await ctx.send(embed=embed)

	# Make and Send Card Analysis
	if with_intention:
		prompt = "My intention in this card pull is: " + intention + "\n\n"
		prompt += "You pulled the " + card_name + " card\n\n"
		prompt += description + "\n\n"
		prompt += "First explain what the intention: '" + intention + "' means, then answer how this intention connects to the card. If it's a question the intention is to know the answer. Write a few sentences and mention the intention directly. Do NOT summarize or repeat the card. Be creative in your interpretation. If the intention is one word talk more about the intention in detail"
		prompt += "\n\n"

		response = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=0.8,
			max_tokens=222,
			top_p=1,
			frequency_penalty=2,
			presence_penalty=2,
			stop=["END"]
		)

		text = response['choices'][0]['text'].strip()
		embed_b = discord.Embed(title = "One Interpretation (beta)", description=text)
		await ctx.send(embed=embed_b)
	
@bot.command(name="ask_group", description="Ask group a question and auto-summarize")
async def ask_group(ctx, *, question=""):

	if len(question) == 0:
		return

	global people
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	users = []

	# Get Birdies
	guild = bot.get_guild(989662771329269890)
	members = discord.utils.get(guild.roles, name="Birdies").members
	birdies = [x.name for x in members]

	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	# Debug Mode
	for guild in bot.guilds:
		for member in guild.members:
			if member.name in birdies:
				users.append(member)

	# Get people in Garden
	responses = []
	views = []
	t_embed = discord.Embed(title = "Time Limit", description = f"Please reply within 45 minutes of receipt")
	i_url = "https://media.discordapp.net/attachments/989662771329269893/1019641048407998464/chrome_Drbki2l0Qq.png"
	c_embed = discord.Embed(title="Confluence Experiment", description = question)
	c_embed.set_image(url=i_url)

	# Message Users
	for person in users:
		view, modal = response_view(modal_text=question)
		try: 
			await person.send(embed=c_embed)
			await person.send(view=view)
			await person.send(embed=t_embed)
		except:
			continue
		responses.append(modal)
		views.append(view)

	# Gather Answers
	all_text = []

	for response in responses:
		await response.wait()
		all_text.append(response.answer.value)

	joined_answers = ""

	for t in all_text:
		if t is not None:
			joined_answers += t + "\n\n"

	if len(joined_answers.strip()) == 0:
		j_embed = discord.Embed(title = "No Responses", description = f"No responses provided to summarize")
		await ctx.send(embed=j_embed)
		return

	if len(responses) == 1:
		k_embed = discord.Embed(title = "One Response", description = all_text[0])
		await ctx.send(embed=k_embed)
		return

	# Chunk Answers
	prompts = []
	answer_chunks = textwrap.wrap(joined_answers, 1750)

	for chunk in answer_chunks:
		prompts.append(pool_prompt(question, joined_answers))

	# Query OpenAI
	response_text = ""

	for prompt in prompts:
		summarized = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=0.6,
			max_tokens=222,
			top_p=1,
			frequency_penalty=1.5,
			presence_penalty=1,
			stop=["END"]
		)
		response_text += summarized.choices[0].text.strip() + "\n\n"
	
	# Send Results to People
	a_embed = discord.Embed(title = "Responses", description = f"{joined_answers}")
	embed = discord.Embed(title = "Consensus (beta)", description = f"**Question**\n{question}\n\n**Consensus**\n{response_text}")

	for person in users:
		try:
			await person.send("Responses", embed=a_embed)
			await person.send("Consensus", embed=embed)
		except:
			continue

	# Send a Redo Option
	for prompt in prompts:
		r_view = redo_view(ctx, prompt, question)
		await ctx.send(view=r_view)

bot.run(discord_key)
