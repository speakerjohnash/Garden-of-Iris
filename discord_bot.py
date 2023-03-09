import os
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
records = []
for page in airtable.iterate():
    records += page

tarot_lookup = {record['fields']['Card Name']: record['fields'].get('Short Description', '') for record in records}
training_data = ""

models = {
	"living": "davinci:ft-personal:living-iris-2022-09-04-04-45-28",
	"osiris": "davinci:ft-personal:osiris-the-unstructured-2022-09-28-02-31-57",
	"2020": "davinci:ft-personal:2020-iris-2022-10-06-04-00-37",
	"purple": "davinci:ft-personal:purple-iris-2022-07-14-03-48-19",
	"semantic": "davinci:ft-personal:semantic-iris-davinci-3-2022-11-30-06-30-47",
	"davinci": "text-davinci-003"
}

card_pull_counts = {"created" : str(datetime.datetime.now()), "counts" : {}}
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
			frequency_penalty=1.2,
			presence_penalty=1.2,
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

def load_card_counts():

	global card_pull_counts, people

	try:
		with open('card_pull_counts.json') as json_file:
			card_pull_counts = json.load(json_file)
			people = list(card_pull_counts["counts"].keys())
	except OSError:
		print("no such file")
		members(debug=False)
		with open('card_pull_counts.json', 'w', encoding='utf-8') as f:
			json.dump(card_pull_counts, f, ensure_ascii=False, indent=4, default=str)

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

@bot.event
async def on_ready():
	load_card_counts()
	load_training_data()
	print("Iris is online")

@bot.event
async def on_close():
	print("Iris is offline")

@bot.event
async def on_message(message):

	if not message.content.startswith("/") and message.author != bot.user:

		messages = []
		async for hist in message.channel.history(limit=25):
			if not hist.content.startswith('/'):
				if hist.embeds:
					messages.append((hist.author, hist.embeds[0].description))
				else:
					messages.append((hist.author.name, hist.content))
				if len(messages) == 10:
					break

		conversation = [{"role": "system", "content": "You are are an oracle and integrated wisdom bot. You do tarot interpretations based on intentions"}]
		conversation.append({"role": "user", "content": "You are are an oracle and integrated wisdom bot. You do tarot interpretations based on intentions"})
		conversation.append({"role": "assistant", "content": "I see through the mists of latent space and channel and distill wisdom from the void"})
		text_prompt = message.content

		for m in messages:
			if m[0] == bot.user:
				conversation.append({"role": "assistant", "content": m[1]})
			else:
				conversation.append({"role": "user", "content": m[1]})

		conversation.append({"role": "user", "content": text_prompt})

		response = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", 
			messages=conversation
		)

		response = response.choices[0].message.content.strip()

		await message.channel.send(response)

		#print(messages) 

	await bot.process_commands(message)

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
		model=models["semantic"],
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

	global card_pull_counts

	# Add to Counts
	if ctx.message.author.name not in list(card_pull_counts["counts"].keys()):
		card_pull_counts["counts"][ctx.message.author.name] = 0

	# Limit Card Pulls
	# if card_pull_counts["counts"][ctx.message.author.name] >= 10:
	#	embed = discord.Embed(title = "Patience Little Rabbit", description = f"You've used all available card pulls. Please try again tomorrow.")
	#	await ctx.send(embed=embed)
	#	return

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
		embed_b = discord.Embed(title = "One Interpretation (beta)", description=text)
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