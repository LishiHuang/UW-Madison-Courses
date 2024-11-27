#%%
import numpy as np
from scipy.optimize import fsolve

def equations(vars, *params):
    k, p0 = vars
    theta, pi = params
    pi0, pi1 = pi
    p1 = 1 - p0
    e0, e1 = 2200, 1700  # Total endowments

    # Equation 1: Market clearing in state 0
    eq1 = e0 - k * e1

    # Equation 2: FOC for homogeneous theta
    eq2 = k ** theta - (pi0 / pi1) * ((1 / p0) - 1)

    return [eq1, eq2]

# Parameters
theta = 1.5
pi = [0.8, 0.2]
params = (theta, pi)

# Initial guesses
k_guess = 1.0
p0_guess = 0.5
initial_guesses = [k_guess, p0_guess]

# Solve the equations
solution = fsolve(equations, initial_guesses, args=params)
k_sol, p0_sol = solution
p1_sol = 1 - p0_sol

print("Part a) Equilibrium Results:")
print(f"k = {k_sol:.4f}")
print(f"p0 = {p0_sol:.4f}")
print(f"p1 = {p1_sol:.4f}")

# Compute individual consumptions
agents = ['Aaron', 'Baron', 'Carlson']
endowments = {
    'Aaron': np.array([500, 400]),
    'Baron': np.array([1000, 500]),
    'Carlson': np.array([700, 800])
}

x_i1 = {}
x_i0 = {}
for agent in agents:
    e_i0, e_i1 = endowments[agent]
    numerator = p0_sol * e_i0 + p1_sol * e_i1
    denominator = p0_sol * k_sol + p1_sol
    x1 = numerator / denominator
    x0 = k_sol * x1
    x_i1[agent] = x1
    x_i0[agent] = x0

    print(f"\n{agent}:")
    print(f"x_{agent}0 = {x0:.4f}")
    print(f"x_{agent}1 = {x1:.4f}")
#%%
# Calculate net demands
for agent in agents:
    e_i0, e_i1 = endowments[agent]
    net_demand_0 = x_i0[agent] - e_i0
    net_demand_1 = x_i1[agent] - e_i1
    print(f"\n{agent} Net Demands:")
    print(f"State 0 Net Demand: {net_demand_0:.4f}")
    print(f"State 1 Net Demand: {net_demand_1:.4f}")
#%%
# Compute stochastic discount factors
pi0, pi1 = pi
m0 = p0_sol / pi0
m1 = p1_sol / pi1

print("\nPart c) Stochastic Discount Factors:")
print(f"m0 = {m0:.4f}")
print(f"m1 = {m1:.4f}")
# %%
# Prices of assets A1 = (2, 3) and A2 = (3, 2)
A1 = [2, 3]
A2 = [3, 2]

p_A1 = p0_sol * A1[0] + p1_sol * A1[1]
p_A2 = p0_sol * A2[0] + p1_sol * A2[1]

print("\nPart d) Prices of Assets:")
print(f"Price of A1 = {p_A1:.4f}")
print(f"Price of A2 = {p_A2:.4f}")
# %%
import numpy as np
from scipy.optimize import fsolve

def equations(vars, *params):
    p0, x_A1, x_B1, x_C1 = vars
    theta_A, theta_B, theta_C, pi = params
    pi0, pi1 = pi
    p1 = 1 - p0
    e0 = 2200
    e1 = 1700
    e_A0, e_A1 = 500, 400
    e_B0, e_B1 = 1000, 500
    e_C0, e_C1 = 700, 800

    # Compute gamma
    gamma = (pi0 * p1) / (pi1 * p0)

    # Compute k_i
    k_A = gamma ** (1 / theta_A)
    k_B = gamma ** (1 / theta_B)
    k_C = gamma ** (1 / theta_C)

    # x_i0 = k_i * x_i1
    x_A0 = k_A * x_A1
    x_B0 = k_B * x_B1
    x_C0 = k_C * x_C1

    # Budget constraints
    eq1 = (p0 * k_A + p1) * x_A1 - (p0 * e_A0 + p1 * e_A1)
    eq2 = (p0 * k_B + p1) * x_B1 - (p0 * e_B0 + p1 * e_B1)
    eq3 = (p0 * k_C + p1) * x_C1 - (p0 * e_C0 + p1 * e_C1)

    # Market clearing in state 1
    eq4 = x_A1 + x_B1 + x_C1 - e1

    return [eq1, eq2, eq3, eq4]

# Parameters
theta_A = 1.5
theta_C = 1.5
pi = [0.8, 0.2]

# Initial guesses
p0_guess = 0.5
x_A1_guess = 400
x_B1_guess = 500
x_C1_guess = 800
initial_guesses = [p0_guess, x_A1_guess, x_B1_guess, x_C1_guess]

theta_B_values = [2.0, 2.5, 3.0]

for theta_B in theta_B_values:
    params = (theta_A, theta_B, theta_C, pi)
    solution = fsolve(equations, initial_guesses, args=params)
    p0_sol, x_A1_sol, x_B1_sol, x_C1_sol = solution
    p1_sol = 1 - p0_sol

    # Compute gamma
    gamma = (pi[0] * p1_sol) / (pi[1] * p0_sol)

    # Compute k_i
    k_A = gamma ** (1 / theta_A)
    k_B = gamma ** (1 / theta_B)
    k_C = gamma ** (1 / theta_C)

    # Compute x_i0
    x_A0_sol = k_A * x_A1_sol
    x_B0_sol = k_B * x_B1_sol
    x_C0_sol = k_C * x_C1_sol

    print(f"\nResults for theta_B = {theta_B}:")
    print(f"p0 = {p0_sol:.4f}")
    print(f"p1 = {p1_sol:.4f}")
    print(f"Aaron's consumption: x_A0 = {x_A0_sol:.4f}, x_A1 = {x_A1_sol:.4f}")
    print(f"Baron's consumption: x_B0 = {x_B0_sol:.4f}, x_B1 = {x_B1_sol:.4f}")
    print(f"Carlson's consumption: x_C0 = {x_C0_sol:.4f}, x_C1 = {x_C1_sol:.4f}")

    # Update initial guesses for next iteration
    initial_guesses = [p0_sol, x_A1_sol, x_B1_sol, x_C1_sol]
