# Iris Discord Bot

The Iris Discord Bot is an interface to cognicism wisdom and ideas. This repository contains a set of training data containing prompt completion pairs describing the functionality of cognicism. It also contains tools for interfacing with wisdom collectively over time. Most of what is contained here is experiments related to the larger vision of cognicism. 

## The most accurate and current representation of cognicism will be found in the chat-iris.csv training data.

## Features

### Frankeniris
The `frankeniris` function combines the outputs of the Iris as imbued with cognicist ideas and the default GPT-4 language model to create a response. It first obtains an answer from the Iris model based on the user's message content. It then constructs a conversation thread for the GPT-4 model using the message history and provides GPT-4 with Iris's answer. Frankeniris aims to provide more creative and integrated responses for the user by combining the knowledge of both models.

### Fourthought Pool
The `fourthought_pool` function assists users in a Discord channel by guiding them through the FourThought dialectic process. The process helps users focus on the future, make predictions, and reflect on the past. The bot provides guidance and adapts to user requests while maintaining a short response length. The bot is triggered whenever a message is sent in the Fourthought Pool channel.

### Iris Pool
More general and simpler access to cognicism knowledge

### Stability Pool
In Progress means to collectively pool resources. Not yet functional.

### Tarot Card Pull (`/pullcard`)
The `pullcard` command allows users to draw a random tarot card with an optional intention. The drawn card, its description, and image are sent to the user. If an intention is provided, the bot also generates and sends an interpretation of the card in the context of the intention using the OpenAI model.

### Ask Iris (`/iris` or `/ask`)
The `iris` command (aliased as `ask`) allows certain users (testers) to directly interact with the Iris language model by providing their thoughts as input. The bot generates a response from the Iris model based on the input, sends the response as an embedded message in Discord, and provides options for users to give feedback, share the response, or request elaboration. The user's input and the model's response are also saved in the 'iris_training-data.csv' file for potential future training.

### Channel Wisdom (`/channel` or `/c`)
The `channel` command (aliased as `c`) allows users to request a snippet of abstract and analytical wisdom related to a specified topic. The function selects a random non-question prompt or completion from a CSV file containing data from previous interactions with the Iris model. It then calls the frankeniris function to generate a creative response based on the selected prompt, which is sent to the user.

### FAQ (`/faq`)
The `faq` command responds with a Frequently Asked Question (FAQ) and its corresponding answer. The bot reads the questions and answers from a CSV file containing the data, selects a random question and its corresponding completion, and sends the question and answer as an embedded message in Discord. It then calls the frankeniris function to provide a more integrated response using the context of the selected FAQ.

### Infuse (`/infuse`, `/in`, `/inject`)
The `infuse` command (aliased as `in` and `inject`) allows users to bring in a source into the stream via scraping and parsing. Users provide a link as input, and the bot processes the link to extract relevant information.

## Getting Started
To use the Iris Discord Bot, you need to have the appropriate API keys for Discord, OpenAI, and Airtable. The bot uses the Discord.py library and the OpenAI API to interact with users and generate responses.

## Commands
The bot supports various commands that users can invoke using the command prefix `/`. Some of the key commands include `/iris`, `/ask`, `/pullcard`, `/ask_group`, `/channel`, `/faq`, and `/infuse`.

## Contributing
Ask John

## Contact
For any questions or inquiries about the Iris Discord Bot, please contact Speaker John Ash
