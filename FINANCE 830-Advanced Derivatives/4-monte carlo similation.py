# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:24:38 2024

@author: Julia
"""

import numpy as np
from scipy.stats import norm

# Given parameters
S0 = K = 5123.79
sigma = 0.16
r = 0.05
T = 1
n_sims = [100, 1000, 10000]
n_trials = 500  # Number of trials for repeating confidence interval calculation


# Black-Scholes formula for European Call Option
def black_scholes_call(S0, K, sigma, r, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2)-S0 * norm.cdf(-d1) 
    print(f"Black-Scholes Price: = {put_price:.4f}")
    return put_price

# Monte Carlo simulation for European Call Option
def monte_carlo_european_call(S0, K, sigma, r, T, n):
    simulations = []
    for _ in range(n):
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.normal(0, 1))
        payoff = max(0, K-ST)
        simulations.append(np.exp(-r * T) * payoff)
    return simulations

# Confidence Interval Calculation
def confidence_interval(mean, std, n):
    z = norm.ppf(0.975)  # 95% confidence interval
    margin_error = z * (std / np.sqrt(n))
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound


black_scholes_price = black_scholes_call(S0, K, sigma, r, T)

# Part (a): Monte Carlo simulations
for n in n_sims:
    option_prices = monte_carlo_european_call(S0, K, sigma, r, T, n)
    mean_price = np.mean(option_prices)
    print(f"For {n} simulations: Mean Option Price = {mean_price:.4f}")

# Part (b): Confidence Intervals using CLT
for n in n_sims:
    option_prices = monte_carlo_european_call(S0, K, sigma, r, T, n)
    mean_price = np.mean(option_prices)
    std_price = np.std(option_prices)
    lower_bound, upper_bound = confidence_interval(mean_price, std_price, n)
    print(f"For {n} simulations: 95% Confidence Interval = ({lower_bound:.4f}, {upper_bound:.4f})")

# Part (c): Check if Black-Scholes price falls into confidence interval (for n = 1000)
count_inside_interval = 0
op=[]
for _ in range(n_trials):
    option_prices = monte_carlo_european_call(S0, K, sigma, r, T, 1000)
    mean_price = np.mean(option_prices)
    std_price = np.std(option_prices)
    lower_bound, upper_bound = confidence_interval(mean_price, std_price, 1000)
    if lower_bound <= black_scholes_price <= upper_bound:
        count_inside_interval += 1

print(f"Black-Scholes price falls into confidence interval {count_inside_interval} times out of {n_trials}.")
# Part (d): Interpretation
desired_accuracy = 0.01  # For two digits accuracy
required_n = (norm.ppf(0.975) * std_price / desired_accuracy)**2
print(f"To achieve two digits accuracy with 95% probability, {int(np.ceil(required_n))} realizations are needed.")
