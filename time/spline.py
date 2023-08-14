import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class SyntheticTimeSeriesGenerator:

	def __init__(self, start_date, end_date, num_points):
		self.start_date = start_date
		self.end_date = end_date
		self.num_points = num_points
		self.data = pd.DataFrame()

	def create_base_pattern(self):
		# Generate key time points spanning 0 to 1
		key_time_points = np.linspace(0, 1, 20)

		# Generate random key values with more locally constrained transitions
		key_values = np.cumsum(np.random.uniform(-0.1, 0.1, 20))
		
		# Normalize key values to fit within [0, 1]
		key_values = (key_values - min(key_values)) / (max(key_values) - min(key_values))
		
		# Adjust the range to fit within [0, 1000]
		key_values *= 1000

		# Create a cubic spline interpolation for the key points
		cubic_spline = CubicSpline(key_time_points, key_values)
		smooth_base_pattern = cubic_spline(np.linspace(0, 1, self.num_points))

		return smooth_base_pattern

	def add_seasonal_variation(self, prices):
		days_in_year = 365.25
		seasonal_pattern = 1 + 0.05 * np.sin(2 * np.pi * np.arange(len(prices)) / days_in_year)
		return prices * seasonal_pattern

	def add_significant_annual_events(self, prices):
		years = np.linspace(0, len(prices), num=int(len(prices) // 365.25))
		event_years = np.random.choice(years, size=5, replace=False)  # choosing 5 years randomly for significant events
		for year in event_years:
			start, end = int(year), int(year + 365.25)
			event_factor = np.random.uniform(0.9, 1.1)  # deviating up to 10% from base pattern
			prices[start:end] *= event_factor
		return prices

	def add_yearly_oscillation(self, prices):
		oscillation = 0.02 * np.sin(10 * 2 * np.pi * np.arange(len(prices)) / 365.25)
		return prices + oscillation

	def add_monthly_trends(self, prices):
		days_in_month = 30.44  # average days in a month
		monthly_trend = 1 + 0.03 * np.sin(2 * np.pi * np.arange(len(prices)) / days_in_month)
		return prices * monthly_trend

	def add_significant_monthly_events(self, prices):
		months = np.linspace(0, len(prices), num=int(len(prices) // (365.25/12)))
		event_months = np.random.choice(months, size=24, replace=False)  # choosing 2 months per year for significant events
		for month in event_months:
			start, end = int(month), int(month + (365.25/12))
			event_factor = np.random.uniform(0.9, 1.1)
			prices[start:end] *= event_factor
		return prices

	def add_daily_variation_within_month(self, prices):
		day_pattern = np.random.choice([0.98, 1.02], size=len(prices))  # randomly assign a surge or drop for each day
		return prices * day_pattern

	def generate(self):
		date_range = pd.date_range(self.start_date, self.end_date, periods=self.num_points)
		prices = self.create_base_pattern()
		
		# Adding yearly perturbations
		prices = self.add_seasonal_variation(prices)
		prices = self.add_significant_annual_events(prices)
		prices = self.add_yearly_oscillation(prices)

		# Adding monthly perturbations
		prices = self.add_monthly_trends(prices)
		prices = self.add_significant_monthly_events(prices)
		prices = self.add_daily_variation_within_month(prices)
		
		self.data = pd.DataFrame({
			'Date': date_range,
			'Close': prices
		})
		return self.data

	def plot(self):
		plt.figure(figsize=(12, 6))
		plt.plot(self.data['Date'], self.data['Close'])
		plt.title('Synthetic Time Series with Yearly Perturbations')
		plt.xlabel('Date')
		plt.ylabel('Close Price')
		plt.grid(True)
		plt.tight_layout()
		plt.show()

# Usage
generator = SyntheticTimeSeriesGenerator(start_date='1920-01-01', end_date='2020-01-01', num_points=365*100)
generator.generate()
generator.plot()
