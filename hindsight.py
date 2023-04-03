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

def create_summary(conversation):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation
            )
            response = response.choices[0].message.content.strip()
            return response
        except openai.error.RateLimitError:
            print("Rate limit error encountered. Retrying in 5 seconds...")
            time.sleep(5)

def split_text_into_chunks(text, max_chunk_size=4000):

    # Calculate the number of chunks needed to evenly distribute the text
    num_chunks = max(1, (len(text) + max_chunk_size - 1) // max_chunk_size)
    
    # Adjust the chunk size to evenly distribute the text across the chunks
    chunk_size = (len(text) + num_chunks - 1) // num_chunks
    
    # Split the text into chunks of the calculated chunk size
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    return text_chunks

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
        text_chunks = split_text_into_chunks(text)

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

def create_weekly_summaries():
    # This function will aggregate and summarize the daily summaries for each week,
    # highlighting key themes and insights for the week.

    def generate_conversation(text, is_part=False):

        prompt = "Write this section of John Ash's thoughts as a story. Do not make anything up."
        prompt += "It's over a Week so say what week you're covering and then explain what he focused on that week. " if not is_part else "It's over part of a Week so explain what he focused on during this part. "
        
        conversation = [
            {"role": "system", "content": prompt},
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

        # Check Cache
        cached_summary = weekly_summaries_df.loc[weekly_summaries_df['Week_Start_Date'] == week_start_date.strftime('%Y-%m-%d')]

        if not cached_summary.empty:
            continue

        # Prepend the week start date to the list of summaries
        week = week.sort_index()
        summaries_list = [f"Week of {week_start_date.strftime('%Y-%m-%d')}:"]
        
        # Loop through each day in the week DataFrame and append the date and summary to the list
        for date, row in week.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            summary_text = row['Summary']
            summaries_list.append(f"Date: {date_str}\n{summary_text}")

        # Join the list into a single string
        week_text = '\n\n'.join(summaries_list)

        # Chunk Text
        text_chunks = split_text_into_chunks(week_text)

        if len(text_chunks) > 1:

            summaries = []

            for chunk in text_chunks:
                sub_conv = generate_conversation(chunk, is_part=True)
                summary_part = create_summary(sub_conv)
                summaries.append(summary_part)

            summary_text = ' '.join([summary.strip() for summary in summaries])
            sub_conv = [{"role": "system", "content": "You stitch summaries about parts of a week into one. You make minor edits to make multiple sections into one."}]
            sub_conv.append({"role": "user", "content": "Write this as one integrated piece. Copy everything and miss nothing: " + summary_text})

            summary = create_summary(sub_conv)

        else:
            conversation = generate_conversation(week_text)
            summary = create_summary(conversation)

        # Get the date of the week in the desired format
        week_date_str = week_start_date.strftime('%B %d, %Y')

        # Prepend the prefix "The Week of" and append the suffix ":"
        week_date_str = "Week of " + week_date_str + ':'

        # Prepend the date and two newline characters to the summary
        summary = week_date_str + '\n\n' + summary

        # Append the new summary to the CSV file
        with open('weekly_summaries.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([week_start_date.strftime('%Y-%m-%d'), summary])

        print("---\n\n")
        print(summary)


def create_monthly_summaries():
    # This function will aggregate and summarize the weekly summaries for each month,
    # focusing on the most important or relevant content for the month.

    # Load weekly_summaries.csv into a DataFrame
    weekly_summaries_df = pd.read_csv('weekly_summaries.csv', parse_dates=['Week_Start_Date'])
    weekly_summaries_df.set_index('Week_Start_Date', inplace=True)
    
    # Group the weekly summaries by month
    monthly_groups = weekly_summaries_df.resample('M')

    # Load monthly_summaries.csv if it exists, otherwise create a new DataFrame and save it
    if os.path.isfile('monthly_summaries.csv'):
        monthly_summaries_df = pd.read_csv('monthly_summaries.csv', parse_dates=['Month_Start_Date'])
    else:
        monthly_summaries_df = pd.DataFrame(columns=['Month_Start_Date', 'Summary'])
        monthly_summaries_df.to_csv('monthly_summaries.csv', index=False)

    # Loop through each month
    for month_start_date, month in monthly_groups:

        # Check Cache
        cached_summary = monthly_summaries_df.loc[monthly_summaries_df['Month_Start_Date'] == month_start_date.strftime('%Y-%m-%d')]

        if not cached_summary.empty:
            continue

        # Prepend the month start date to the list of summaries
        month = month.sort_index()
        summaries_list = [f"Month of {month_start_date.strftime('%B %Y')}:"]
        
        # Loop through each week in the month DataFrame and append the date and summary to the list
        for date, row in month.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            summary_text = row['Summary']
            summaries_list.append(f"Week of {date_str}:\n{summary_text}")

        # Join the list into a single string
        month_text = '\n\n'.join(summaries_list)

        # Prepend concise user instruction to month_text
        instruction = f"Summarize John Ash's key thoughts and their connection to relevant world events for the month of {month_start_date.strftime('%B %Y')}."
        month_text = instruction + '\n\n' + month_text

        # Generate conversation and summary for the month
        conversation = [
            {"role": "system", "content": "(Background Context: John Ash is a machine learning engineer, musician and artist who specializes in language models like GPT. He is the steward of Cognicism, Iris, Social Proof of Impact and the Prophet Incentive. We are currently scanning through summaries of his thoughts and predictions and mapping them to a real world timeline)"},
            {"role": "system", "content": f"What occurred the month of {month_start_date.strftime('%B %Y')} in the world relevant to John's focus? Start by summarizing the events that occurred in the world that month in two sentences that are relevant to John's focus. You'll have to pull the world events from your own knowledge store, not here. Reference wikipedia as a timeline. Choose events related to his focus and anything mentioned in the summaries below. Focus the story by connecting it to real world events that are related to the story."},
            {"role": "system", "content": "Then write the story arc of John's month and don't make anything up. Form a central focus and narrative through time rather than just summarizing a list of things he thought about. Mention how his thoughts might relate to world events."},
            {"role": "system", "content": "(If there was news related to philosophy, systems change, machine learning and music at the time connect it but don't specifically mention that list of topics)"},
            {"role": "system", "content": "Mention the context of the times and how his thoughts traced a pathway through the era."},
            {"role": "user", "content": month_text}
        ]

        summary = create_summary(conversation)

        # Append the new summary to the CSV file
        with open('monthly_summaries.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([month_start_date.strftime('%Y-%m-%d'), summary])

        print("---\n\n")
        print(summary)

def create_seasonal_summaries():
    # This function will aggregate and summarize the monthly summaries for each season (winter, spring, summer, fall),
    # providing an overview of the key trends and patterns observed during that period.

    # Load monthly_summaries.csv into a DataFrame
    monthly_summaries_df = pd.read_csv('monthly_summaries.csv', parse_dates=['Month_Start_Date'])
    monthly_summaries_df.set_index('Month_Start_Date', inplace=True)

    # Define custom date ranges for each season based on approximate equinoxes and solstices
    seasons = {
        'Winter': ('12-21', '03-20'),
        'Spring': ('03-21', '06-20'),
        'Summer': ('06-21', '09-22'),
        'Fall': ('09-23', '12-20')
    }

    # Loop through each season
    for season, (start_day, end_day) in seasons.items():
        # Create a boolean mask to select rows within the date range for each season
        mask = ((monthly_summaries_df.index.month == int(start_day.split('-')[0])) & (monthly_summaries_df.index.day >= int(start_day.split('-')[1]))) | \
               ((monthly_summaries_df.index.month == int(end_day.split('-')[0])) & (monthly_summaries_df.index.day <= int(end_day.split('-')[1])))

        # Apply the mask to the DataFrame to get the data for the current season
        season_data = monthly_summaries_df[mask]

        # Generate summary for the current season
        season_text = f"Season of {season}:\n\n" + '\n\n'.join(season_data['Summary'].tolist())

        # Generate conversation and summary for the season
        conversation = [
            {"role": "system", "content": "Summarize John Ash's key thoughts and their connection to relevant world events for the season of " + season + ". Form a central focus and narrative through time rather than just summarizing a list of things he thought about. Mention how his thoughts might relate to world events."},
            {"role": "user", "content": season_text}
        ]

        summary = create_summary(conversation)

        print("---\n\n")
        print(summary)

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
#daily_summaries = create_daily_summaries(df_merged)
weekly_summaries = create_weekly_summaries()