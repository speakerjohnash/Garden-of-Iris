import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define time range
start_date = datetime(1922, 1, 1)
end_date = datetime(2022, 1, 1)

# Variable time range generation
def generate_irregular_dates(start_date, end_date, num_points):
    dates = []
    total_days = (end_date - start_date).days
    avg_interval = total_days / num_points
    current_date = start_date
    while len(dates) < num_points:
        dates.append(current_date)
        interval = np.random.normal(avg_interval, avg_interval / 5) 
        current_date += timedelta(days=int(interval))
    return dates[:num_points]

# Number of data points to generate
num_points = 5000
dates = generate_irregular_dates(start_date, end_date, num_points)

def generate_dynamic_pattern(dates):
    pattern = np.zeros(len(dates))
    idx = 0

    while idx < len(dates):
        seasonalities = {
            'minute': np.random.randint(2, 10),
            'hourly': np.random.randint(2, 10),
            'daily': np.random.randint(2, 10),
            'weekly': np.random.randint(2, 10),
            'monthly': np.random.randint(2, 10),        
            'yearly': np.random.randint(2, 10)
        }

        segment_length = np.random.randint(30, 500)  # Random segment length
        end_idx = min(idx + segment_length, len(dates))

        segment = (
            seasonalities['minute'] * np.sin(2*np.pi*np.array([date.minute for date in dates[idx:end_idx]])/60) +
            seasonalities['hourly'] * np.sin(2*np.pi*np.array([date.hour for date in dates[idx:end_idx]])/24) +
            seasonalities['daily'] * np.sin(2*np.pi*np.array([date.day for date in dates[idx:end_idx]])/31) +
            seasonalities['weekly'] * np.sin(2*np.pi*np.array([date.weekday() for date in dates[idx:end_idx]])/7) + 
            seasonalities['monthly'] * np.sin(2*np.pi*np.array([date.month for date in dates[idx:end_idx]])/12) +
            seasonalities['yearly'] * np.sin(2*np.pi*np.array([date.timetuple().tm_yday for date in dates[idx:end_idx]])/365)
        )
        
        pattern[idx:end_idx] = segment
        idx = end_idx

    return pattern

# Generate the series with dynamic oscillation patterns
series = generate_dynamic_pattern(dates)

# Add noise
noise = np.random.normal(0, 0.1, len(series))  
series += noise

# Convert to DataFrame with appropriate column naming
df = pd.DataFrame({
    'Date': dates,
    'Close': series
})

# Add some one-off events
event_dates = np.random.choice(df.index, size=10, replace=False)
for idx in event_dates:
    df.loc[idx, 'Close'] += np.random.uniform(1, 5)

# Export to CSV
df.to_csv('synthetic_data.csv', index=False)
