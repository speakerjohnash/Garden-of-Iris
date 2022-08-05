import os

import openai
import discord

from discord.ui import Button, View, TextInput
from discord.ext import commands

discord_key = os.getenv("DISCORD_BOT_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

@bot.event
async def on_ready():
	print("Iris is online")

@bot.command()
async def ping(ctx):
	await ctx.send('Pong!')

def members(ctx):
	members = []
	for guild in bot.guilds:
		for member in guild.members:
			members.append(member)
	unique_members = [*set(members)]
	return unique_members

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
async def claim(ctx, thought=""):
	"""
	/claim log a claim for the iris to learn
	"""

	button = Button(label="button label", style=discord.ButtonStyle.blurple)

	async def button_callback(interaction):
		await interaction.response.send_message(thought)

	button.callback = button_callback

	view = View()
	view.add_item(button)
	
	await ctx.send(thought, view=view)

@bot.command()
async def ask_group(ctx, *, question=""):
	"""
	/ask_group ask group a question and have davinci summarize
	"""

	# Get people in Garden
	people = members(ctx)

	# Message Users

	for person in people:
		if person.name != "Golden Iris":
			print(person.name)
			await person.send(question)

	# Gather Answers

	"""

	joined_answers, summarized = "", ""
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
	
	await ctx.send(summarized) */

	"""

bot.run(discord_key)