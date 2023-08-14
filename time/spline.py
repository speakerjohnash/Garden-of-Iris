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

    def generate(self):
        date_range = pd.date_range(self.start_date, self.end_date, periods=self.num_points)
        prices = self.create_base_pattern()
        self.data = pd.DataFrame({
            'Date': date_range,
            'Close': prices
        })
        return self.data

    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Date'], self.data['Close'])
        plt.title('Synthetic Time Series')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Usage
generator = SyntheticTimeSeriesGenerator(start_date='1920-01-01', end_date='2020-01-01', num_points=365*100)
generator.generate()
generator.plot()
