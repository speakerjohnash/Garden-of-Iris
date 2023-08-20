import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from scipy.interpolate import CubicSpline

class SyntheticTimeSeriesGenerator:

	def __init__(self, start_date, end_date, mode='default'):
		self.start_date = pd.to_datetime(start_date)
		self.end_date = pd.to_datetime(end_date)
		self.date_range = pd.date_range(self.start_date, self.end_date, freq='T') # Minute frequency
		self.num_points = len(self.date_range)
		self.data = pd.DataFrame()
		self.mode = mode

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

	def add_weekly_trends(self, prices):
		days_in_week = 7	# days in a week
		# Using a sinusoidal function to create a smooth weekly pattern
		weekly_trend = 1 + 0.02 * np.sin(2 * np.pi * np.arange(len(prices)) / days_in_week)
		return prices * weekly_trend

	def add_seasonal_variation(self, prices):
		days_in_year = 365.25
		seasonal_pattern = 1 + 0.05 * np.sin(2 * np.pi * np.arange(len(prices)) / days_in_year)
		return prices * seasonal_pattern

	def add_significant_annual_events(self, prices):
		years = np.linspace(0, len(prices), num=int(len(prices) // 365.25))
		event_years = np.random.choice(years, size=5, replace=False)	# choosing 5 years randomly for significant events
		for year in event_years:
			start, end = int(year), int(year + 365.25)
			event_factor = np.random.uniform(0.9, 1.1)	# deviating up to 10% from base pattern
			prices[start:end] *= event_factor
		return prices

	def add_yearly_oscillation(self, prices):
		oscillation = 0.02 * np.sin(10 * 2 * np.pi * np.arange(len(prices)) / 365.25)
		return prices + oscillation

	def add_monthly_trends(self, prices):
		days_in_month = 30.44	# average days in a month
		monthly_trend = 1 + 0.03 * np.sin(2 * np.pi * np.arange(len(prices)) / days_in_month)
		return prices * monthly_trend

	def add_significant_monthly_events(self, prices):
		months = np.linspace(0, len(prices), num=int(len(prices) // (365.25/12)))
		event_months = np.random.choice(months, size=24, replace=False)	# choosing 2 months per year for significant events
		for month in event_months:
			start, end = int(month), int(month + (365.25/12))
			event_factor = np.random.uniform(0.9, 1.1)
			prices[start:end] *= event_factor
		return prices

	def add_daily_variation_within_month(self, prices):
		day_pattern = np.random.choice([0.98, 1.02], size=len(prices))	# randomly assign a surge or drop for each day
		return prices * day_pattern

	def generate(self):

		mode = self.mode

		# Call complex data generation
		if mode == 'complex':
			return self.generate_complex()
		
		# Call simple sine wave generation
		elif mode == 'simple':	
			return self.generate_simple()

		elif mode == 'default':	
			return self.generate_complex()

		else:
			raise ValueError("Invalid mode specified")

	# Complex Data Generation
	def generate_complex(self):
		date_range = self.date_range
		prices = self.create_base_pattern()
		
		# Adding yearly perturbations
		prices = self.add_seasonal_variation(prices)
		prices = self.add_significant_annual_events(prices)
		prices = self.add_yearly_oscillation(prices)

		# Adding monthly perturbations
		prices = self.add_monthly_trends(prices)
		prices = self.add_significant_monthly_events(prices)
		prices = self.add_daily_variation_within_month(prices)
		
		# Adding weekly perturbations
		prices = self.add_weekly_trends(prices)

		self.data = pd.DataFrame({
			'Date': date_range,
			'Close': prices
		})

		return self.data

	# Simple sine wave generation method
	def generate_simple(self, num_cycles=9):

		# Calculate sine wave
		time_delta = self.end_date - self.start_date
		ang_freq = 2 * np.pi * num_cycles / time_delta.total_seconds() * 60 # Considering minute frequency

		values = np.sin(ang_freq * np.arange(self.num_points))
		values = (values + 1) * 500

		# Create dataframe
		self.data = pd.DataFrame({
				'Date': self.date_range,
				'Close': values
		})

		return self.data

	def save_to_csv(self, max_points=None):
		data = self.data
		if max_points is not None and len(data) > max_points:
			data = data.sample(n=max_points).sort_values('Date')
		data.to_csv('synthetic_data.csv', index=False)

	def plot(self):

		def plot_range(ax, title, start_idx, duration):
			data = self.data.iloc[start_idx:start_idx + duration]
			if len(data) > 1000:
				data = data.sample(n=1000).sort_index()
			ax.plot(data['Date'], data['Close'])
			ax.set_title(title)
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
			ax.grid(True)

		fig = plt.figure(figsize=(18, 12))

		plot_data = [
			('Full Range', 0, len(self.data)),
			('Random Decade', np.random.randint(0, len(self.data) - 365 * 24 * 60 * 10), 365 * 24 * 60 * 10) if len(self.data) >= 365 * 24 * 60 * 10 else None,
			('Random Year', np.random.randint(0, len(self.data) - 365 * 24 * 60), 365 * 24 * 60) if len(self.data) >= 365 * 24 * 60 else None,
			('Random Month', np.random.randint(0, len(self.data) - 30 * 24 * 60), 30 * 24 * 60) if len(self.data) >= 30 * 24 * 60 else None,
			('Random Week', np.random.randint(0, len(self.data) - 7 * 24 * 60), 7 * 24 * 60) if len(self.data) >= 7 * 24 * 60 else None,
			('Random Day', np.random.randint(0, len(self.data) - 24 * 60), 24 * 60) if len(self.data) >= 24 * 60 else None
		]

		plot_data = [item for item in plot_data if item is not None]

		for i, (title, start_idx, duration) in enumerate(plot_data):
			ax = fig.add_subplot(3, 2, i + 1)
			plot_range(ax, title, start_idx, duration)

		plt.tight_layout()
		plt.show()

# Usage
generator = SyntheticTimeSeriesGenerator(start_date='2008-01-01', end_date='2019-01-15', mode="simple")
generator.generate()
generator.plot()
generator.save_to_csv(max_points=20000)
