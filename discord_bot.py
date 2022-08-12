import os
import random

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

	async def on_submit(self, interaction: discord.Interaction):
		embed = discord.Embed(title = self.title, description = f"**{self.answer.label}**\n{self.answer}")
		embed.set_author(name = interaction.user)
		await interaction.response.send_message(embed=embed)

	async def on_timeout(self):
		self.stop()

def button_view(modal_text="default text"):

	modal = AskModal(title=modal_text)

	async def button_callback(interaction):
		answer = await interaction.response.send_modal(modal)

	view = View()
	view.timeout = 90.0
	button = Button(label="Answer", style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)

	return view, modal

def members(ctx):

	members = []

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
async def ask(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	print(thought)

	response = openai.Completion.create(
		model="text-davinci-002",
		prompt=thought,
		temperature=1,
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

	view = button_view()

	await ctx.send(thought, view=view)

@bot.command()
async def pullcard(ctx, *, intention=""):

	with_intention = len(intention) > 0
	# r_num = random.random()
	# if with_intention: random.seed(intention + str(r_num))

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
		prompt += "First explain what the intention means, then answer how this intention connects to the card. If it's a question the intention is to know the answer. Write a few sentences and mention the intention directly. Do NOT summarize or repeat the card. Be creative in your interpretation"
		prompt += "\n\n"

		response = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=0.8,
			max_tokens=128,
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
	people = members(ctx)
	responses = []

	# Message Users
	for person in people:
		view, modal = button_view(modal_text=question)
		responses.append(modal)
		await person.send(question, view=view)

	# Gather Answers
	all_text = []		

	for response in responses:
		timed_out = await response.wait()
		print(timed_out)
		print(response.answer)
		if not timed_out:
			all_text.append(response.answer)

	print("when do we arrive here?")

	joined_answers = ""

	for t in all_text:
		joined_answers += t.value + "\n"

	prompt = question + "\n" + joined_answers + "\nWhat is the consensus above?"

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
	
	embed = discord.Embed(title = question, description = f"**Answer**\n{response_text}")
	await ctx.send(embed=embed)

	#await ctx.send(summarized)

bot.run(discord_key)