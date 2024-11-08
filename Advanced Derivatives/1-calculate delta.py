import pandas as pd
import numpy as np
from scipy.stats import norm

# Read the Excel file
df = pd.read_excel("FI830_HW4Series-2.xlsx")

# Extract necessary columns
actual_prices = df["Actual"]
fict_1_prices = df["Fict. 1"]
fict_2_prices = df["Fict. 2"]
fict_3_prices = df["Fict. 3"]
df['Day_to_expiration'] = range(252)

# Set up parameters
K = 3977.19  # strike price
sigma = 0.12  # volatility
r = 0.05  # risk-free interest rate
Tt = 1  # time to expiration (in years)

# Initialize lists to store deltas
deltas_actual = []
deltas_fict1 = []
deltas_fict2 = []
deltas_fict3 = []

# Calculate delta for each day for each path
for actual_price, fict_1_price, fict_2_price, fict_3_price, time in zip(actual_prices, fict_1_prices, fict_2_prices, fict_3_prices, df['Day_to_expiration']):
    T = time / 252
    
    d1_actual = (np.log(actual_price / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    delta_actual = norm.cdf(d1_actual)
    deltas_actual.append(delta_actual)
    
    d1_fict1 = (np.log(fict_1_price / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    delta_fict1 = norm.cdf(d1_fict1)
    deltas_fict1.append(delta_fict1)
    
    d1_fict2 = (np.log(fict_2_price / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    delta_fict2 = norm.cdf(d1_fict2)
    deltas_fict2.append(delta_fict2)
    
    d1_fict3 = (np.log(fict_3_price / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    delta_fict3 = norm.cdf(d1_fict3)
    deltas_fict3.append(delta_fict3)

# Create a DataFrame to store deltas
df['Delta Actual'] = deltas_actual
df["Delta Fict. 1"] = deltas_fict1
df["Delta Fict. 2"] = deltas_fict2
df["Delta Fict. 3"] = deltas_fict3

# Save the DataFrame to an Excel file
df.to_excel("delta_values_corrected.xlsx", index=False)
