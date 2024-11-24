# %%
import numpy as np
from scipy.optimize import fmin_cobyla
import matplotlib.pyplot as plt

# %%
import numpy as np
from scipy.optimize import fmin_cobyla

# Endowments
e_A0 = 40  # Agent A's endowment of good 1
e_A1 = 25  # Agent A's endowment of good 2
e_B0 = 60  # Agent B's endowment of good 1
e_B1 = 45  # Agent B's endowment of good 2
alpha_A = np.array([3, 4])  # Agent A's utility function coefficients
alpha_B = np.array([2, 1])  # Agent B's utility function coefficients

# Total endowments
e_0 = e_A0 + e_B0
e_1 = e_A1 + e_B1

# Prices
p_0 = 1  # Price of good 1 (fixed)
p_1 = 0.75  # Initial price guess for good 2

# Negative of the utility function (to minimize)
def U_A(x):
    return -(alpha_A[0] * np.log(x[0]) + alpha_A[1] * np.log(x[1]))

def U_B(x):
    return -(alpha_B[0] * np.log(x[0]) + alpha_B[1] * np.log(x[1]))

# Compute wealth for both agents
def wealth(p_1):
    m_A = p_0 * e_A0 + p_1 * e_A1  # Agent A's wealth
    m_B = p_0 * e_B0 + p_1 * e_B1  # Agent B's wealth
    return m_A, m_B

m_A, m_B = wealth(p_1)

# Budget constraint for Agent A
def budget_A(x):
    return m_A - x[0] * p_0 - x[1] * p_1

# Budget constraint for Agent B
def budget_B(x):
    return m_B - x[0] * p_0 - x[1] * p_1

# Non-negativity constraints for goods
def x_0_positive(x):
    return x[0]

def x_1_positive(x):
    return x[1]

# Solve the optimization problem for Agent A
solution_A = fmin_cobyla(
    U_A,  # Objective function (negative utility)
    [1, (m_A - p_0) / p_1],  # Initial guess (for x1 and x2)
    [budget_A, x_0_positive, x_1_positive],  # Constraints
    rhoend=1e-7  # Precision of the solver
)

# Solve the optimization problem for Agent B
solution_B = fmin_cobyla(
    U_B,  # Objective function (negative utility)
    [1, (m_B - p_0) / p_1],  # Initial guess (for x1 and x2)
    [budget_B, x_0_positive, x_1_positive],  # Constraints
    rhoend=1e-7  # Precision of the solver
)

# Total demand in the market (Agent A's demand + Agent B's demand)
demand = solution_A + solution_B

print("Agent A demand is : ", solution_A)
print("Agent B demand is : ", solution_B)
print("Total demand is : ", demand)

# Net demand: total demand minus total endowments
net_demand = demand - [e_0, e_1]

# Print the net demand result
print("Net demand is: ", net_demand)


# %%
# Create a linspace to search the optimal p_1
n_points = 150
p_1_range = np.linspace(1, 1.5, n_points)

# Let's store both agents demand of each good
A0_demand = np.zeros(np.size(p_1_range))
A1_demand = np.zeros(np.size(p_1_range))
B0_demand = np.zeros(np.size(p_1_range))
B1_demand = np.zeros(np.size(p_1_range))
demand_0 = np.zeros(np.size(p_1_range))
demand_1 = np.zeros(np.size(p_1_range))
net_demands = np.zeros(np.size(p_1_range))


# Iterate over all prices
for i in range(n_points):
    
    p_1 = p_1_range[i]
    
    # Initial wealth for both agents at current prices
    m_A, m_B = wealth(p_1)

    # Solve the optimization problem for Agent A
    solution_A = fmin_cobyla(
        U_A,  # Objective function (negative utility)
        [1, (m_A - p_0) / p_1],  # Initial guess (for x1 and x2)
        [budget_A, x_0_positive, x_1_positive],  # Constraints
        rhoend=1e-7  # Precision of the solver
    )

    # Solve the optimization problem for Agent B
    solution_B = fmin_cobyla(
        U_B,  # Objective function (negative utility)
        [1, (m_B - p_0) / p_1],  # Initial guess (for x1 and x2)
        [budget_B, x_0_positive, x_1_positive],  # Constraints
        rhoend=1e-7  # Precision of the solver
    )

    # Total demand in the market (Agent A's demand + Agent B's demand)
    demand = solution_A + solution_B
    net_demand = demand - [e_0, e_1]

    # Store the obtained values 
    A0_demand[i] = solution_A[0]
    A1_demand[i] = solution_A[1]
    B0_demand[i] = solution_B[0]
    B1_demand[i] = solution_B[1]
    demand_0[i] = demand[0]
    demand_1[i] = demand[1]
    # For the net demand we store the sum of squares
    # note that we want this sum to be as close as zero as 
    # possible.
    net_demands[i] = np.sum( net_demand ** 2)

equilibrium_price = p_1_range[ np.argmin(net_demands) ]
print("Equilibrium price =", equilibrium_price)
print("Agent A demand is : ", A0_demand[np.argmin(net_demands)], A1_demand[np.argmin(net_demands)])
print("Agent B demand is : ", B0_demand[np.argmin(net_demands)], B1_demand[np.argmin(net_demands)])

# Plot the net demands curve
plt.plot(p_1_range, net_demands, label='Net Demand')

# Add a vertical line for the equilibrium price
plt.axvline(equilibrium_price, color='red', linestyle='--', label=f'Equilibrium Price (p1 = {equilibrium_price:.2f})')

# Add labels for the axes
plt.xlabel('Price of Good 1 (p1)')
plt.ylabel('Net Demand')

# Add a title
plt.title('Net Demand Curve and Equilibrium Price')

# Add a grid for clarity
plt.grid(True)

# Add a legend
plt.legend()

# Display the plot
plt.show()


# %%

# Define the total endowment of goods 0 and 1
total_e0 = e_0
total_e1 = e_1

# Define grid for plotting
A0_vals = np.linspace(0, total_e0, 100)
A1_vals = np.linspace(0, total_e1, 100)
A0_mesh, A1_mesh = np.meshgrid(A0_vals, A1_vals)

# Compute utilities using your predefined utility functions U_A and U_B
U_A_vals = np.zeros_like(A0_mesh)
U_B_vals = np.zeros_like(A0_mesh)

# Fill utility values for plotting indifference curves
for i in range(len(A0_vals)):
    for j in range(len(A1_vals)):
        U_A_vals[i, j] = U_A([A0_mesh[i, j], A1_mesh[i, j]])  # Agent A's utility
        U_B_vals[i, j] = U_B([A0_mesh[i, j], A1_mesh[i, j]])  # Agent B's utility

# Plot the Edgeworth box
fig, ax = plt.subplots(figsize=(8, 8))

# Agent A's equilibrium demand
A0_eq = A0_demand[np.argmin(net_demands)]
A1_eq = A1_demand[np.argmin(net_demands)]

# Agent B's equilibrium demand (rest of the total endowment)
B0_eq = total_e0 - A0_eq
B1_eq = total_e1 - A1_eq

# Plot the origins and allocations
ax.plot(0, 0, 'ro', label='Agent A Origin (O_A)', markersize=10)
ax.plot(total_e0, total_e1, 'bo', label='Agent B Origin (O_B)', markersize=10)
ax.plot(A0_eq, A1_eq, 'go', label='Equilibrium (x)', markersize=10)

# Draw the boundary of the Edgeworth box
ax.plot([0, total_e0], [total_e1, total_e1], 'k--')  # Top boundary
ax.plot([0, total_e0], [0, 0], 'k--')  # Bottom boundary
ax.plot([0, 0], [0, total_e1], 'k--')  # Left boundary
ax.plot([total_e0, total_e0], [0, total_e1], 'k--')  # Right boundary

# Get utility values for the equilibrium point for both agents
UA_eq = U_A([A0_eq, A1_eq])
UB_eq = U_B([B0_eq, B1_eq])

# Define indifference levels, ensuring they are strictly increasing
indifference_levels_A = sorted([UA_eq * 0.8, UA_eq, UA_eq * 1.2])
indifference_levels_B = sorted([UB_eq * 0.8, UB_eq, UB_eq * 1.2])

# Plot indifference curves for Agent A (Orange)
ax.contour(A0_mesh, A1_mesh, U_A_vals, levels=indifference_levels_A, colors='red')

# For Agent B, we need to invert the axes because the origin is in the top right corner
# Invert the values of Agent B's meshgrid to reflect this
B0_mesh = total_e0 - A0_mesh  # Flip A0 mesh for Agent B
B1_mesh = total_e1 - A1_mesh  # Flip A1 mesh for Agent B

# Compute the utility values for Agent B based on the flipped mesh
U_B_vals = np.array([[U_B([B0_mesh[i, j], B1_mesh[i, j]]) for j in range(B1_mesh.shape[1])] for i in range(B0_mesh.shape[0])])

# Plot indifference curves for Agent B (Orange, inverted for B)
ax.contour(A0_mesh, A1_mesh, U_B_vals, levels=indifference_levels_B, colors='orange')


# Plot the Pareto efficiency curve (Green) - here, just connecting the equilibrium
pareto_x = [0, A0_eq, total_e0]
pareto_y = [0, A1_eq, total_e1]
ax.plot(pareto_x, pareto_y, 'g-', label='Pareto Efficiency Curve')

# Labels and titles
ax.set_xlabel('Good 0 (Agent A → Right, Agent B ← Left)')
ax.set_ylabel('Good 1 (Agent A → Up, Agent B ← Down)')
ax.set_title('Edgeworth Box with Indifference Curves and Pareto Efficiency')

# Axis limits
ax.set_xlim(0, total_e0)
ax.set_ylim(0, total_e1)

# Grid and legend
ax.grid(True)
ax.legend()

# Show the plot
plt.show()


# %%
# Create a linspace to search the optimal p_1
n_points = 150
p_1_range = np.linspace(0.5, 1.5, n_points)

e_A0 = 40  # Agent A's endowment of good 1
e_A1 = 35  # Agent A's endowment of good 2
e_B0 = 60  # Agent B's endowment of good 1
e_B1 = 55  # Agent B's endowment of good 2
alpha_A = np.array([3, 4])  # Agent A's utility function coefficients
alpha_B = np.array([2, 1])  # Agent B's utility function coefficients

# Total endowments
e_0 = e_A0 + e_B0
e_1 = e_A1 + e_B1


# Let's store both agents demand of each good
A0_demand = np.zeros(np.size(p_1_range))
A1_demand = np.zeros(np.size(p_1_range))
B0_demand = np.zeros(np.size(p_1_range))
B1_demand = np.zeros(np.size(p_1_range))
demand_0 = np.zeros(np.size(p_1_range))
demand_1 = np.zeros(np.size(p_1_range))
net_demands = np.zeros(np.size(p_1_range))


# Iterate over all prices
for i in range(n_points):
    
    p_1 = p_1_range[i]
    
    # Initial wealth for both agents at current prices
    m_A, m_B = wealth(p_1)

    # Solve the optimization problem for Agent A
    solution_A = fmin_cobyla(
        U_A,  # Objective function (negative utility)
        [1, (m_A - p_0) / p_1],  # Initial guess (for x1 and x2)
        [budget_A, x_0_positive, x_1_positive],  # Constraints
        rhoend=1e-7  # Precision of the solver
    )

    # Solve the optimization problem for Agent B
    solution_B = fmin_cobyla(
        U_B,  # Objective function (negative utility)
        [1, (m_B - p_0) / p_1],  # Initial guess (for x1 and x2)
        [budget_B, x_0_positive, x_1_positive],  # Constraints
        rhoend=1e-7  # Precision of the solver
    )

    # Total demand in the market (Agent A's demand + Agent B's demand)
    demand = solution_A + solution_B
    net_demand = demand - [e_0, e_1]

    # Store the obtained values 
    A0_demand[i] = solution_A[0]
    A1_demand[i] = solution_A[1]
    B0_demand[i] = solution_B[0]
    B1_demand[i] = solution_B[1]
    demand_0[i] = demand[0]
    demand_1[i] = demand[1]
    # For the net demand we store the sum of squares
    # note that we want this sum to be as close as zero as 
    # possible.
    net_demands[i] = np.sum( net_demand ** 2)

equilibrium_price = p_1_range[ np.argmin(net_demands) ]
print("Equilibrium price =", equilibrium_price)
print("Agent A demand is : ", A0_demand[np.argmin(net_demands)], A1_demand[np.argmin(net_demands)])
print("Agent B demand is : ", B0_demand[np.argmin(net_demands)], B1_demand[np.argmin(net_demands)])

# Plot the net demands curve
plt.plot(p_1_range, net_demands, label='Net Demand')

# Add a vertical line for the equilibrium price
plt.axvline(equilibrium_price, color='red', linestyle='--', label=f'Equilibrium Price (p1 = {equilibrium_price:.2f})')

# Add labels for the axes
plt.xlabel('Price of Good 1 (p1)')
plt.ylabel('Net Demand')

# Add a title
plt.title('Net Demand Curve and Equilibrium Price')

# Add a grid for clarity
plt.grid(True)

# Add a legend
plt.legend()

# Display the plot
plt.show()


