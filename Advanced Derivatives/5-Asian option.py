

import numpy as np
from scipy.stats import norm

# Given parameters
S0 = K = 3977.19
sigma = 0.12
r = 0.05
T = 1
n= 10000

def monte_carlo_stock_path(S0, K, sigma, r, dt):
    path=[]
    for _ in range(252):
        S0=S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1))
        path.append(S0)
    return np.mean(path)

# Monte Carlo simulation for European Call Option
def monte_carlo_european_call(S0, K, sigma, r, T, n):
    dt=T/252
    simulations = []
    for _ in range(n):
        A = monte_carlo_stock_path(S0, K, sigma, r, dt)
        payoff = max(0, A - K)
        simulations.append(np.exp(-r * T) * payoff)
    return simulations

# Confidence Interval Calculation
def confidence_interval(mean, std, n):
    z = norm.ppf(0.975)  # 95% confidence interval
    margin_error = z * (std / np.sqrt(n))
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound

# Part (a): Monte Carlo simulations

option_prices = monte_carlo_european_call(S0, K, sigma, r, T, n)
mean_price = np.mean(option_prices)
std_price = np.std(option_prices)
print(f"For {n} simulations: Mean Option Price = {mean_price:.4f}")

