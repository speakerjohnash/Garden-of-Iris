import os
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

df = pd.read_csv('tarot_text.csv')
names = df['card_name'].tolist()
descriptions = df['text'].tolist()
airtable = Table(airtable_key, 'app2X00KuiIxPwGsf', 'cards')
tarot_lookup = dict(zip(names, descriptions))

card_pull_counts = {"created" : str(datetime.datetime.now()), "counts" : {}}
people = []

class AskModal(Modal, title="Ask Modal"):

	answer = TextInput(label="Answer", max_length=256, style=discord.TextStyle.long)

	def add_view(self, question, view: View):
		self.answer.placeholder = question[0:100]
		self.view = view

	async def on_submit(self, interaction: discord.Interaction):
		embed = discord.Embed(title = "Your Response", description = f"\n{self.answer}")
		embed.set_author(name = interaction.user)
		await interaction.response.send_message(embed=embed)
		self.view.stop()

def button_view(modal_text="default text"):	

	async def view_timeout():
		modal.stop()	

	view = View()
	view.on_timeout = view_timeout
	view.timeout = 3600.0
	view.auto_defer = False

	modal = AskModal(title="Response")
	modal.auto_defer = False
	modal.timeout = 3600.0

	async def button_callback(interaction):
		answer = await interaction.response.send_modal(modal)

	button = Button(label="Answer", style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)
	modal.add_view(modal_text, view)

	return view, modal

def load_card_counts():

	global card_pull_counts, people

	try:
		with open('card_pull_counts.json') as json_file:
			card_pull_counts = json.load(json_file)
			people = list(card_pull_counts["counts"].keys())
	except OSError:
		print("no such file")
		members(debug=False)

	# Reset After a Day
	reset_card_counts()

def reset_card_counts():

	global card_pull_counts

	# Reset After a Day
	created = dateutil.parser.parse(card_pull_counts["created"])
	now = datetime.datetime.now()
	time_passed = now - created

	if time_passed.seconds >= 86400:
		card_pull_counts = {"created" : datetime.datetime.now(), "counts" : {}}
		members(debug=False)
		with open('card_pull_counts.json', 'w', encoding='utf-8') as f:
			json.dump(card_pull_counts, f, ensure_ascii=False, indent=4, default=str)

def members(debug=False):

	members = []
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	global card_pull_counts, people

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
	names = [member.name for member in unique_members]
	counts = dict(zip(names, [0]*len(names)))
	card_pull_counts['counts'] = counts
	people = unique_members

def redo_view(ctx, prompt, question):

	async def button_callback(interaction):

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

def make_prompt(question, joined_answers):

	prompt = question + "\n\nAnswers are below"
	prompt += "\n\nWrite a long detail paragraph summarizing and analyzing the answers below. What are the commonalities and differences in the answers?"
	prompt += "\n\n---\n\nAnswers:\n\n" + joined_answers
	prompt += "---\n\nThe question was: " + question + "\n\n"
	prompt += "Write a long detailed paragraph about the question as if you were a singular voice formed from all of the views above. What does this community believe?"
	prompt += "\n\nSum the answers into one answer that best represents all the views shared. If many questions are provided respond with a question representing what most people are uncertain about"
	prompt += "\n\n"

	return prompt

@bot.event
async def on_ready():
	load_card_counts()
	print("Iris is online")

@bot.event
async def on_close():
	print("Iris is offline")

@bot.command()
async def ask(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	
	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	response = openai.Completion.create(
		model="davinci:ft-personal:purple-iris-2022-07-14-03-48-19",
		prompt=thought,
		temperature=0.69,
		max_tokens=114,
		top_p=1,
		frequency_penalty=1.46,
		presence_penalty=1.54,
		stop=["END"]
	)

	text = response['choices'][0]['text']

	await ctx.send(text.strip())

@bot.command()
async def claim(ctx, *, thought=""):
	"""
	/claim log a claim for the iris to learn
	"""

	# TODO
	# Save claim to datastore

	await ctx.send(thought)

@bot.command()
async def pullcard(ctx, *, intention=""):

	global card_pull_counts

	# Only Allow Some Users
	if ctx.message.author.name not in list(card_pull_counts["counts"].keys()):
		return

	# Limit Card Pulls
	if card_pull_counts["counts"][ctx.message.author.name] >= 3:
		embed = discord.Embed(title = "Patience Little Rabbit", description = f"You've used all available card pulls. Please try again tomorrow.")
		await ctx.send(embed=embed)
		return

	# With Intention
	with_intention = len(intention) > 0
	r_num = random.random()
	if with_intention: random.seed(intention + str(r_num))

	# Get Embed Data
	card_name = random.choice(list(tarot_lookup.keys()))
	description = tarot_lookup[card_name].strip()

	url = ""

	for page in airtable.iterate():
		for record in page:
			if record["fields"]["Card Name"] == card_name:
				if "All images" in record["fields"]:
					url = record["fields"]["All images"][0]["url"]

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
		embed_b = discord.Embed(title = "One Interpretation", description=text)
		await ctx.send(embed=embed_b)

	# Update Card Pull Counts
	card_pull_counts["counts"][ctx.message.author.name] += 1
	reset_card_counts()

	with open('card_pull_counts.json', 'w', encoding='utf-8') as f:
		json.dump(card_pull_counts, f, ensure_ascii=False, indent=4, default=str)
	
@bot.command(name="ask_group", description="Ask group a question and auto-summarize")
async def ask_group(ctx, *, question=""):

	if len(question) == 0:
		return

	global people
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	users = []

	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	# Debug Mode
	for guild in bot.guilds:
		for member in guild.members:
			if member.name in testers:
				users.append(member)

	# Get people in Garden
	responses = []
	views = []
	t_embed = discord.Embed(title = "Time Limit", description = f"Please reply within 60 minutes of receipt. We do this so we can collect the data in timely manner and deliver it to people.")
	i_url = "https://media.discordapp.net/attachments/989662771329269893/1019641048407998464/chrome_Drbki2l0Qq.png"
	c_embed = discord.Embed(title="Confluence Experiment", description = question)
	c_embed.set_image(url=i_url)

	# Message Users
	for person in users:
		view, modal = button_view(modal_text=question)
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
		r_embed = discord.Embed(title = "No Responses", description = f"No responses provided to summarize")
		await ctx.send(embed=r_embed)
		return

	# Chunk Answers
	prompts = []
	answer_chunks = textwrap.wrap(joined_answers, 1750)

	for chunk in answer_chunks:
		prompts.append(make_prompt(question, joined_answers))

	# Query OpenAI
	response_text = ""

	for prompt in prompts:
		summarized = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=0.7,
			max_tokens=222,
			top_p=1,
			frequency_penalty=2,
			presence_penalty=0.7,
			stop=["END"]
		)
		response_text += summarized.choices[0].text.strip()
	
	# Send Results to People
	a_embed = discord.Embed(title = "Responses", description = f"{joined_answers}")
	embed = discord.Embed(title = "Consensus", description = f"**Question**\n{question}\n\n**Consensus**\n{response_text}")

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