# %%
import numpy as np
from scipy.optimize import fmin_cobyla

# Given data
A = np.array([[0.10, 0.14, 1.25],
              [0.10, 1.14, 0.00],
              [1.10, 0.00, 0.00]])
p = np.array([1.0, 1.1, 1.2])
pi = np.array([0.3, 0.5, 0.2])
m = 1000  # Initial wealth

# Part a) Compute Arrow security prices
p_bar = np.linalg.solve(A.T, p)

# Part b) Optimize expected utility subject to budget constraint
def expected_utility(x):
    return -np.dot(pi, np.log(x))  # Negative for maximization

def budget_constraint(x):
    return m - np.dot(p_bar, x)  # Budget constraint

# Initial guess for optimal wealth distribution
x0 = np.array([m / 3] * 3)

# Constraints
constraints = [lambda x: x[i] for i in range(3)] + [budget_constraint]

# Optimize
optimal_wealth = fmin_cobyla(expected_utility, x0, constraints)

# Part c) Find portfolio of assets (A0, A1, A2)
q = np.linalg.solve(A, optimal_wealth)

# Display results
print("Arrow security prices:", p_bar)
print("Optimal wealth distribution:", optimal_wealth)
print("Portfolio of assets:", q)


# %%
# Given data
A = np.array([[0.6, 0.0, 1.0],
              [0.4, 1.0, 0.0],
              [0.0, 0.0, 1.0]])
p = np.array([1.2, 1.2, 0.6])
pi = np.array([0.4, 0.4, 0.2])
m = 1000  # Initial wealth

# Part a) Compute Arrow security prices
p_bar = np.linalg.solve(A.T, p)

# Part b) Optimize expected utility subject to budget constraint
def expected_utility(x):
    return -np.dot(pi, np.log(x))  # Negative for maximization

def budget_constraint(x):
    return m - np.dot(p_bar, x)  # Budget constraint

# Initial guess for optimal wealth distribution
x0 = np.array([m / 3] * 3)

# Constraints
constraints = [lambda x: x[i] for i in range(3)] + [budget_constraint]

# Optimize
optimal_wealth = fmin_cobyla(expected_utility, x0, constraints)

# Part c) Find portfolio of assets (A0, A1, A2)
q = np.linalg.solve(A, optimal_wealth)

# Display results
print("Arrow security prices:", p_bar)
print("Optimal wealth distribution:", optimal_wealth)
print("Portfolio of assets:", q)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Given values
price_riskless = 1    # Price of riskless asset (A0)
price_risky = 10      # Price of risky asset (A1)
initial_wealth = 1_000_000  # Ben's wealth in case of no flood

# Calculate the possible wealth allocations in the flood and no-flood states based on budget constraint
# Set up points for the constraint line where wealth in no-flood state changes from 0 to max wealth
no_flood_wealth = np.linspace(0, initial_wealth, 100)  # Wealth in no-flood state
flood_wealth = (initial_wealth - price_risky * no_flood_wealth) / price_riskless

# Plotting the constraint line
plt.figure(figsize=(8, 6))
plt.plot(no_flood_wealth, flood_wealth, label='Budget Constraint', color='blue')

# Adding labels and title
plt.xlabel("Wealth in No-Flood State ($x_1$)")
plt.ylabel("Wealth in Flood State ($x_0$)")
plt.title("Ben's Budget Constraint Line")
plt.legend()
plt.grid(True)

plt.show()



