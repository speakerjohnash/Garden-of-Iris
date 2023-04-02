import os
import re
import csv
import sys
import time
import random
import pandas as pd
from datetime import datetime
from itertools import islice
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def create_summary(conversation):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.42,
                messages=conversation
            )
            response = response.choices[0].message.content.strip()
            return response
        except openai.error.RateLimitError:
            print("Rate limit error encountered. Retrying in 5 seconds...")
            time.sleep(5)

def create_weekly_summaries():
    # This function will aggregate and summarize the daily summaries for each week,
    # highlighting key themes and insights for the week.

    def generate_conversation(text, is_part=False):

        prompt = "You read in a sequence of John Ash's thoughts from a week and summarize what he thinking. These thoughts are predictive. Note any accurate predictions"
        prompt += "Write John Ash's thoughts as a story. Do not make anything up."
        prompt += "It's over a Week so say what week you're covering and then explain what he focused on that week. " if not is_part else "It's over part of a Week so explain what he focused on during this part. "
        
        conversation = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": "The thoughts you are receiving are from the past. You can validate the predictions with current information. Specifically say which predictions have been validated by time. The current date and time is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"role": "system", "content": "Form a narrative. Find a through thread of his focus through this time. Be very detailed but don't share unncessary tangents. Do not share anything indicated to be private. Note any predictions he got particularly right"},
            {"role": "user", "content": text}
        ]

        return conversation

    daily_summaries_df = pd.read_csv('daily_summaries.csv', parse_dates=['Date'])
    daily_summaries_df.set_index('Date', inplace=True)
    weekly_groups = daily_summaries_df.resample('W-MON')

    # Load weekly_summaries.csv if it exists, otherwise create a new DataFrame and save it
    if os.path.isfile('weekly_summaries.csv'):
        weekly_summaries_df = pd.read_csv('weekly_summaries.csv', parse_dates=['Week_Start_Date'])
    else:
        weekly_summaries_df = pd.DataFrame(columns=['Week_Start_Date', 'Summary'])
        weekly_summaries_df.to_csv('weekly_summaries.csv', index=False)

    # Loop through each week
    for week_start_date, week in weekly_groups:

        # Prepend the week start date to the list of summaries
        summaries_list = [f"Week of {week_start_date.strftime('%Y-%m-%d')}:"]
        summaries_list.extend(week['Summary'].tolist())
        week_text = '\n\n'.join(summaries_list)

        # Chunk Text
        text_chunks = [week_text[i:i+4000] for i in range(0, len(week_text), 4000)]

        if len(text_chunks) > 1:

            summaries = []

            for chunk in text_chunks:
                sub_conv = generate_conversation(chunk, is_part=True)
                summary_part = create_summary(sub_conv)
                summaries.append(summary_part)

            summary_text = ' '.join([summary.strip() for summary in summaries])
            sub_conv = [{"role": "system", "content": "You stitch summaries about parts of a week into one. You remove unncessary line breaks and make minor edits to make multiple sections into one."}]
            sub_conv.append({"role": "user", "content": "Write this as one paragraph with no unnecessary line breaks. Copy everything and miss nothing: " + summary_text})

            summary = create_summary(sub_conv)

        else:
            conversation = generate_conversation(week_text)
            summary = create_summary(conversation)

        # Append the new summary to the CSV file
        with open('weekly_summaries.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([week_start_date.strftime('%Y-%m-%d'), summary])

        print("---\n\n")
        print(summary)

def create_monthly_summaries(df):
    # This function will aggregate and summarize the daily or weekly summaries for each month,
    # focusing on the most important or relevant content for the month.
    pass

def create_seasonal_summaries(df):
    # This function will aggregate and summarize the monthly summaries for each season (quarter),
    # providing an overview of the key trends and patterns observed during that period.
    pass

def create_yearly_summaries(df):
    # This function will aggregate and summarize the monthly or seasonal summaries for each year,
    # capturing the most significant themes and learnings for the year.
    pass

def create_certainty_summaries(df):
    # This function will calculate the average certainty for different time periods
    # (e.g., daily, weekly, monthly) and provide summaries of the general level of certainty
    # for each period.
    pass

def create_sentiment_summaries(df):
    # This function will calculate the average sentiment (valence) for different time periods
    # and provide summaries of the general sentiment (positive or negative) for each period.
    pass

def create_temporal_focus_summaries(df):
    # This function will group thoughts by categories such as thought type (Reflect, Ask, Predict, State)
    # and calculate the average temporal focus (past, present, future) for each category over
    # a specified time period.
    pass

def create_trend_analysis(df):
    # This function will identify recurring themes, topics, or patterns over time and provide
    # summaries of notable trends observed in the data. It may use word counts.
    pass

def create_periodic_reflections(df):
    # This function will summarize key learnings, reflections, and takeaways for specific random time periods
    pass

def create_predictions_review(df):
    # This function will summarize predictions made during a specific time period and, if possible,
    # provide an evaluation of their accuracy based on subsequent outcomes.
    pass

def create_trackable_summaries(df):
    # This function will extract and organize trackable data (numeric variables that change over time)
    # from the text using a consistent format (e.g., "#weight: 175 lbs"). It will calculate summary
    # statistics for each trackable and generate summaries that describe the changes in trackables
    # over time, including any notable trends, patterns, or associations with other events.
    pass

def load_and_preprocess_csv(csv_file):

    df = pd.read_csv(csv_file)
    df['Post date'] = pd.to_datetime(df['Post date'], format='%m/%d/%y %I:%M %p').dt.tz_localize('US/Pacific')

    # Split the 'Good' column into positive and negative values
    split_good = df['Good'].str.split('\n', expand=True)
    df['Positive'] = split_good[0]
    df['Negative'] = split_good[1]

    # Convert 'Positive' and 'Negative' columns to numeric
    df['Positive'] = pd.to_numeric(df['Positive'].str.replace('+', '', regex=False))
    df['Negative'] = pd.to_numeric(df['Negative'].str.replace('-', '').apply(lambda x: '-' + x))

    # Drop the original 'Good' column
    df.drop(columns=['Good'], inplace=True)
    df['Positive'] = df['Positive'].astype("Int64")
    df['Negative'] = df['Negative'].astype("Int64")

    return df

def create_daily_summaries(df):
    # This function will create summaries of thoughts, reflections, questions, and predictions
    # for each day, providing a brief overview of the main topics discussed on that day.

    # Group the data by date (daily) and include all columns for each group
    df.set_index('Post date', inplace=True)
    daily_groups = df.groupby(df.index.date)

    # islice(daily_groups, 20)

    # Load daily_summaries.csv if it exists, otherwise create a new DataFrame and save it
    if os.path.isfile('daily_summaries.csv'):
        summary_df = pd.read_csv('daily_summaries.csv', parse_dates=['Date'])
    else:
        summary_df = pd.DataFrame(columns=['Date', 'Summary'])
        summary_df.to_csv('daily_summaries.csv', index=False)

    for day_date, day in daily_groups:
        
        # Check if there's already a cached summary for the day
        cached_summary = summary_df.loc[summary_df['Date'] == day_date.strftime('%Y-%m-%d')]

        if not cached_summary.empty:
            # print(f"Using cached summary for {day_date.strftime('%Y-%m-%d')}: {cached_summary['Summary'].values[0]}")
            continue

        truth_scale = "0: \"100 percent certain false\"\n25: \"Moderate certainty of falsity\"\n50: \"Complete uncertainty\"\n75: \"Moderate certainty of truth\"\n100: \"100 percent certain true\""

        conversation = [
            {"role": "system", "content": "You read in a sequence of John Ash's thoughts from a day and summarize what he thinking about that day"},
            {"role": "system", "content": "These thoughts are either from Twitter or have metadata from a dialectic called fourthought to help contextualize the flow of cognition"},
            {"role": "system", "content": "In Fourthought, each thought is tagged with a thought type: prediction, statements, reflection, or question"},
            {"role": "system", "content": "Predictions, reflections and statements are about the future, past and present respectively. Each has two voting systems: truth and good"},
            {"role": "system", "content": "Truth is measured on a scale from 0 to 100. In the absence of any votes, no certainty level has been provided. Here is the scale: " + truth_scale},
            {"role": "system", "content": "Sentiment is calculated from good votes and bad votes and is on a scale of -1, 0, 1 with 0 indicating neutrality"},
            {"role": "system", "content": "You are receiving timestamps and can make inferences about the length of time between thoughts as to whether they're connected"},
            {"role": "system", "content": "The thoughts you are receiving are from the past. The current date and time is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            #{"role": "system", "content": "Thoughts with the pattern #trackable: [value] are repeating values that a user is tracking. If there is no unit usually the trackable is on a scale of ten. #mood is an similar to American grade scaling with a 7.5 incidating an average mood"}
        ]

        text = f"Date: {day_date.strftime('%Y-%m-%d')}\n"

        day = day.sort_index()

        for index, row in day.iterrows():

            platform = row['Platform']
            any_twitter = False

            if platform == 'fourthought':

                sentiment_votes = abs(row['Positive']) + abs(row['Negative'])
                positivity = "N/A" if sentiment_votes == 0 else (abs(row['Positive']) - abs(row['Negative'])) / sentiment_votes

                # Manual Alterations for Privacy
                thought_text = row['Thought']

                # Add Privacy information
                privacy_status = "Public" if row['Privacy'] == 0 else "Private"

                text += f"Dialectic: {platform}\n"
                text += f"Timestamp: {index.strftime('%m/%d/%y %I:%M %p')}\n"
                text += f"Thought: {thought_text}\n"
                text += f"Sentiment: " + str(positivity) + "\n"
                text += f"Good Votes: {row['Positive']}\n"
                text += f"Bad Votes: {abs(row['Negative'])}\n"
                text += f"Average Certainty: {row['Truth']}\n"
                text += f"Privacy: {privacy_status}\n"
                text += f"Speaker: John Ash\n"
                text += f"Thought Type: {row['Type']}\n\n"

            elif platform == 'twitter':

                any_twitter = True
                tweet_text = row['full_text']
                retweet_count = row['retweet_count']
                favorite_count = row['favorite_count']
        
                text += f"Platform: {platform}\n"
                text += f"Timestamp: {index.strftime('%m/%d/%y %I:%M %p')}\n"
                text += f"Tweet: {tweet_text}\n"
                text += f"Retweet Count: {retweet_count}\n"
                text += f"Favorite Count: {favorite_count}\n\n"

        # Print the concatenated text for each day
        conversation.append({"role": "system", "content": "Only say what is in the text itself. Be careful about summarizing private thoughts"})

        # Split the text into chunks of 4000 characters or less
        text_chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]

        # Do Recursive Summarization if Too Long
        if len(text_chunks) > 1:

            summaries = []

            for chunk in text_chunks:
                sub_conv = conversation[:]
                sub_conv.append({"role": "user", "content": "You are summarizing part of a day this time. Please state the range of the day (TIME to TIME) that you're summarizing and then summarize what this speaker was thinking about during this section of the day (and reference the period of the day TIME to TIME) into a paragraph story in relation to their place in time. Reference anything in the dialectic helpful towards telling that story but don't make anything up. Be detailed and reference every thought: " + chunk})
                response = create_summary(sub_conv)
                summaries.append(response)

            summary_text = '\n\n'.join([summary.strip() for summary in summaries])
            sub_conv = [{"role": "system", "content": "You stitch summaries about parts of a day into one. You remove line breaks and make minor edits to make multiple sections into one. You mostly copy text"}]
            sub_conv.append({"role": "user", "content": "Write this as one paragraph with no line breaks. Copy everything and miss nothing: " + summary_text})
            response = create_summary(sub_conv)

        else:
            conversation.append({"role": "user", "content": "Please summarize what this speaker was thinking about into a story in relation to their place in time. Reference anything in the dialectic helpful towards telling that story but don't make anything up. Be detailed and reference every thought: " + text})
            response = create_summary(conversation)

        # Append the new summary to the CSV file
        with open('daily_summaries.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([day_date.strftime('%Y-%m-%d'), response])

        print(response + "\n")

# Load and preprocess the CSV into a DataFrame
df = load_and_preprocess_csv('prophet_thought_dump_ALL_THOUGHTS_2023.csv')

# Load fourthought labeled thoughts
df_thoughts = load_and_preprocess_csv('prophet_thought_dump_ALL_THOUGHTS_2023.csv')
df_thoughts['Platform'] = 'fourthought'

df_tweets = pd.read_csv('twitter_archive.csv')
df_tweets['Platform'] = 'twitter'

# Rename 'created_at' column to 'Post date' in df_tweets to match df_thoughts
df_tweets.rename(columns={'created_at': 'Post date'}, inplace=True)

# Drop the 'lang' column from df_tweets
df_tweets.drop(columns=['lang'], inplace=True)

# Convert 'Post date' column to datetime in df_tweets
df_tweets['Post date'] = pd.to_datetime(df_tweets['Post date'], format='%a %b %d %H:%M:%S %z %Y').dt.tz_convert('US/Pacific')
df_tweets['retweet_count'] = df_tweets['retweet_count'].astype("Int64")
df_tweets['favorite_count'] = df_tweets['favorite_count'].astype("Int64")

# Merge the two dataframes based on 'Post date'
df_merged = pd.concat([df_thoughts, df_tweets], axis=0, ignore_index=True, sort=False)

# Sort the merged dataframe by 'Post date'
df_merged.sort_values(by='Post date', inplace=True)
df_merged.reset_index(drop=True, inplace=True)

# Generate daily summaries using the preprocessed DataFrame
# daily_summaries = create_daily_summaries(df_merged)
weekly_summaries = create_weekly_summaries()