import json
import pandas as pd
import re

# Load the tweet.js file
with open('data/tweets.js', 'r', encoding='utf-8') as file:
    content = file.read()
    json_start = content.find('[')
    json_end = content.rfind(']') + 1
    content = content[json_start:json_end]

try:
    tweets = json.loads(content)
except json.decoder.JSONDecodeError as e:
    print("Error:", e)
    print("Surrounding content:")
    print(content[e.pos - 50:e.pos + 50])

print(tweets[0:10])

# Extract relevant information from the tweets
data = []
for tweet in tweets:
    data.append({
        'id': tweet['tweet']['id'],
        'created_at': tweet['tweet']['created_at'],
        'full_text': tweet['tweet']['full_text'],
        'retweet_count': tweet['tweet']['retweet_count'],
        'favorite_count': tweet['tweet']['favorite_count'],
        'lang': tweet['tweet']['lang'],
    })

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv('twitter_archive.csv', index=False)

print("Twitter archive has been successfully converted to CSV.")