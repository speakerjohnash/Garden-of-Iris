from datetime import datetime
from math import sin, cos, pi

def encode_time(start_date, claim_date):
    # Extracting the year difference
    year_difference = claim_date.year - start_date.year

    # Extracting date components
    minute = claim_date.minute
    hour = claim_date.hour
    day = claim_date.day
    month = claim_date.month

    # Calculations
    minute_enc = [sin(2 * pi * minute / 60), cos(2 * pi * minute / 60)]
    hour_enc = [sin(2 * pi * hour / 24), cos(2 * pi * hour / 24)]
    day_enc = [sin(2 * pi * day / 31), cos(2 * pi * day / 31)]
    month_enc = [sin(2 * pi * month / 12), cos(2 * pi * month / 12)]
    year_enc = [sin(2 * pi * year_difference / 1000), cos(2 * pi * year_difference / 1000)]

    # Stacked encoding
    stacked_enc = [
        minute_enc,
        hour_enc,
        day_enc,
        month_enc,
        year_enc,
    ]

    return stacked_enc

# Example start and claim dates
start_date = datetime(1776, 7, 4) # Example start date for seed point
claim_date = datetime(1863, 8, 19, 11, 2) # Example date and time for the claim point

# Test the function
encoded_values = encode_time(start_date, claim_date)

# Print the results
print("Stacked encoding:")
for encoding in encoded_values:
    print(encoding)