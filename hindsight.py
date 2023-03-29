import os
import re
import pandas as pd
from datetime import datetime
from itertools import islice
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def create_weekly_summaries(df):
    # This function will aggregate and summarize the daily summaries for each week,
    # highlighting key themes and insights for the week.
    pass

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
    df['Post date'] = pd.to_datetime(df['Post date'], format='%m/%d/%y %I:%M %p')

    # Split the 'Good' column into positive and negative values
    split_good = df['Good'].str.split('\n', expand=True)
    df['Positive'] = split_good[0]
    df['Negative'] = split_good[1]

    # Convert 'Positive' and 'Negative' columns to numeric
    df['Positive'] = pd.to_numeric(df['Positive'].str.replace('+', '', regex=False))
    df['Negative'] = pd.to_numeric(df['Negative'].str.replace('-', '').apply(lambda x: '-' + x))

    # Drop the original 'Good' column
    df.drop(columns=['Good'], inplace=True)
    
    return df

def create_daily_summaries(df):
    # This function will create summaries of thoughts, reflections, questions, and predictions
    # for each day, providing a brief overview of the main topics discussed on that day.

    def create_summary(conversation):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        response = response.choices[0].message.content.strip()
        return response

    # Group the data by date (daily) and include all columns for each group
    df.set_index('Post date', inplace=True)
    daily_groups = df.groupby(df.index.date)

    summaries = []

    for day_date, day in islice(daily_groups, 20):
        # Extract the date from the index of the DataFrame daily_groups

        conversation = [
            {"role": "system", "content": "You read in a sequence of John Ash's thoughts from a day and summarize what he thinking about that day"},
            {"role": "system", "content": "These thoughts have metadata from a dialectic called fourthought to help contextualize the flow of cognition"},
            {"role": "system", "content": "Each thought is tagged with a thought type: prediction, statements, reflection, or question"},
            {"role": "system", "content": "Predictions, reflections and statements are about the future, past and present respectively. Each has two voting systems: truth and good"},
            {"role": "system", "content": "Truth is a scale of 0 to 100. 50 means uncertain. Anything below 50 means the thought is leaning false, anything over means leaning true. 75 is a medium level of certainty of truth. 25 is a medium level of certainty of falsity. If there is no vote, no one provided any certainty vote"},
            {"role": "system", "content": "You are receiving timestamps and can make inferences about the length of time between thoughts as to whether they're connected"}
        ]

        text = f"Date: {day_date.strftime('%Y-%m-%d')}\n"

        day = day.sort_index()

        for index, row in day.iterrows():

            sentiment_votes = row['Positive'] + row['Negative']
            positivity = " " if sentiment_votes == 0 else row['Positive'] / sentiment_votes

            thought_text = row['Thought']

            text += f"Timestamp: {index.strftime('%m/%d/%y %I:%M %p')}\n"
            text += f"Thought: {thought_text}\n"
            text += f"Sentiment: " + str(positivity) + "\n"
            text += f"Good Votes: {row['Positive']}\n"
            text += f"Bad Votes: {row['Negative']}\n"
            text += f"Average Certainty: {row['Truth']}\n"
            text += f"Speaker: John Ash\n"
            text += f"Thought Type: {row['Type']}\n\n"

        # Print the concatenated text for each day
        conversation.append({"role": "system", "content": "Only say what is in the text itself"})

        # Split the text into chunks of 4000 characters or less
        text_chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]

        # Do Recursive Summarization if Too Long
        if len(text_chunks) > 1:

            summaries = []

            for chunk in text_chunks:
                sub_conv = conversation[:]
                sub_conv.append({"role": "user", "content": "You are summarizing part of a day this time. Please state the range of the day (TIME to TIME) that you're summarizing and then summarize what this speaker was thinking about during this section of the day (and reference the period of the day TIME to TIME) into a paragraph story in relation to their place in time. Reference anything in the dialectic helpful towards telling that story but don't make anything up: " + chunk})
                response = create_summary(sub_conv)
                summaries.append(response)

            summary_text = '\n\n'.join([summary.strip() for summary in summaries])
            sub_conv = [{"role": "system", "content": "You stitch summaries about parts of a day into one. You remove line breaks and make minor edits to make multiple sections into one. You mostly copy text"}]
            sub_conv.append({"role": "user", "content": "Write this as one paragraph with no line breaks. Copy everything and miss nothing: " + summary_text})
            response = create_summary(sub_conv)

        else:
            conversation.append({"role": "user", "content": "Please summarize what this speaker was thinking about into a story in relation to their place in time. Reference anything in the dialectic helpful towards telling that story but don't make anything up: " + text})
            response = create_summary(conversation)

        print(response + "\n")

# Load and preprocess the CSV into a DataFrame
df = load_and_preprocess_csv('prophet_thought_dump_ALL_THOUGHTS_2023.csv')

# Generate daily summaries using the preprocessed DataFrame
daily_summaries = create_daily_summaries(df)