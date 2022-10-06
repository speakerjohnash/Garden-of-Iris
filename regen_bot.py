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

discord_key = os.getenv("CERES_DISCORD_BOT_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

training_data = ""
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

def elaborate(ctx, prompt="prompt"):

	e_prompt = prompt + ". \n\n More thoughts in detail below. \n\n"

	button = Button(label="elaborate", style=discord.ButtonStyle.blurple)

	async def button_callback(interaction):

		if button.disabled:
			return

		button.disabled = True
		await interaction.response.defer()

		response = openai.Completion.create(
			model="davinci:ft-personal:ceres-a-2022-07-01-02-30-41",
			prompt=e_prompt,
			temperature=0.22,
			max_tokens=222,
			top_p=1,
			frequency_penalty=2,
			presence_penalty=2,
			stop=["END"]
		)

		response_text = response.choices[0].text.strip()

		if len(response_text) == 0:

			response = openai.Completion.create(
				model="davinci:ft-personal:ceres-a-2022-07-01-02-30-41",
				prompt=e_prompt,
				temperature=0.8,
				max_tokens=222,
				top_p=1,
				frequency_penalty=1.7,
				presence_penalty=1.7,
				stop=["END"]
			)

			response_text = response.choices[0].text.strip()

		response_text = response_text.replace("###", "").strip()

		if len(response_text) == 0: response_text = "Ceres has no more to communicate after two requests"

		embed = discord.Embed(title = "Elaboration (beta)", description = f"**Prompt**\n{prompt}\n\n**Elaboration**\n{response_text}")

		await ctx.send(embed=embed)


	button.callback = button_callback

	return button

def load_training_data():

	global training_data

	try:
		training_data = pd.read_csv('ceres_training-data.csv')
	except:
		with open('ceres_training-data.csv', 'w', encoding='utf-8') as f:
			training_data = pd.DataFrame(columns=['prompt', 'completion', 'speaker'])
			training_data.to_csv('ceres_training-data.csv', encoding='utf-8', index=False)

@bot.event
async def on_ready():
	load_training_data()
	print("Ceres is online")

@bot.event
async def on_close():
	print("Ceres is offline")

@bot.command(aliases=['ask'])
async def ceres(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	global training_data
	testers = ["John Ash's Username for Discord", "Gregory | RND"]
	
	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	thought_prompt = thought + "\n\n###\n\n"

	response = openai.Completion.create(
		model="davinci:ft-personal:ceres-a-2022-07-01-02-30-41",
		prompt=thought_prompt,
		temperature=0.69,
		max_tokens=222,
		top_p=1,
		frequency_penalty=1.8,
		presence_penalty=1.5,
		stop=["END"]
	)

	text = response['choices'][0]['text']
	text = text.replace("###", "").strip()
	embed = discord.Embed(title = "", description=f"**Prompt**\n{thought}\n\n**Response**\n{text}")

	await ctx.send(embed=embed)

	# Send Clarification and Share UI
	view, modal = response_view(modal_text="Write your clarification here", modal_label="Clarification", button_label="feedback")
	el_prompt = thought + "\n\n" + text
	elaborate_button = elaborate(ctx, prompt=el_prompt)
	view.add_item(elaborate_button)
	await ctx.send(view=view)

	# Save Clarification
	await modal.wait()

	prompt = thought + "\n\n" + text

	if modal.answer.value is not None:
		training_data.loc[len(training_data.index)] = [prompt, modal.answer.value, ctx.message.author.name] 
		training_data.to_csv('ceres_training-data.csv', encoding='utf-8', index=False)

@bot.command()
async def claim(ctx, *, thought):
	"""
	/claim log a claim for the iris to learn
	"""

	global training_data

	# Send Clarification and Share UI
	prompt = "Share something about in the general latent space of Regen Network"

	if thought is not None:
		training_data.loc[len(training_data.index)] = [prompt, thought, ctx.message.author.name] 
		training_data.to_csv('ceres_training-data.csv', encoding='utf-8', index=False)

	await ctx.send("Attestation saved")

@bot.command()
async def davinci(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	global training_data
	testers = ["John Ash's Username for Discord", "Gregory | RND"]
	
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

	await ctx.send(embed=embed)
	await ctx.send(view=view)

bot.run(discord_key)