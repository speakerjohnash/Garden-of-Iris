import os
import random
import time

import openai
import discord
import asyncio
import aiohttp
import json

import pandas as pd

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

class AskModal(Modal, title="Ask Modal"):

	answer = TextInput(label="Answer")

	def add_view(self, view: View):
		self.view = view

	async def on_submit(self, interaction: discord.Interaction):
		embed = discord.Embed(title = "Your Response", description = f"**Question**\n{self.title}\n\n**{self.answer.label}**\n{self.answer}")
		embed.set_author(name = interaction.user)
		await interaction.response.send_message(embed=embed)
		self.view.stop()

def button_view(modal_text="default text"):	

	async def view_timeout():
		modal.stop()	

	view = View()
	view.on_timeout = view_timeout
	view.timeout = 60.0
	view.auto_defer = False

	modal = AskModal(title=modal_text)
	modal.auto_defer = False
	modal.timeout = 60.0

	async def button_callback(interaction):
		answer = await interaction.response.send_modal(modal)

	button = Button(label="Answer", style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)
	modal.add_view(view)

	return view, modal

def members(ctx, debug=False):

	members = []
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]

	if debug:
		for guild in bot.guilds:
			for member in guild.members:
				if member.name in testers:
					members.append(member)
	else:
		for guild in bot.guilds:
			for member in guild.members:
				if member.name != "Golden Iris":
					members.append(member)

	unique_members = [*set(members)]

	return unique_members

@bot.event
async def on_ready():
	print("Iris is online")

@bot.command()
async def asko(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

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


@bot.command(
	name="ask_group",
	description="Ask group a question and have davinci summarize"
)
async def ask_group(ctx, *, question=""):

	# Get people in Garden
	people = members(ctx, debug=True)
	responses = []
	views = []

	# Message Users
	for person in people:
		view, modal = button_view(modal_text=question)
		await person.send(question, view=view)
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
			joined_answers += t + "\n"

	prompt = question + "\n\nAnswers:\n" + joined_answers + "\nWhat is the general consensus above? Are there any notable outliers. How are the opinions similar or different? What further questions could be asked to the group for more clarification?\n\n###\n"

	summarized = openai.Completion.create(
		model="text-davinci-002",
		prompt=prompt,
		temperature=0.7,
		max_tokens=128,
		top_p=1,
		frequency_penalty=2,
		presence_penalty=2,
		stop=["END"]
	)

	response_text = summarized.choices[0].text.strip()
	
	a_embed = discord.Embed(title = "Responses", description = f"{joined_answers}")
	embed = discord.Embed(title = "Consensus", description = f"**Question**\n{question}\n\n**Consensus**\n{response_text}")
	await ctx.send(embed=a_embed)
	await ctx.send(embed=embed)

bot.run(discord_key)