# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.97  # Discount factor
theta = 2    # Risk aversion parameter (commonly used value)
T = 99       # Time horizon
r = 0.03     # Interest rate

# Trader's income for different periods
e_t = np.zeros(T+1)
e_t[0:35] = 0.5   # Income from t=0 to t=34
e_t[35:65] = 1.0  # Income from t=35 to t=64
# Retirement period income is zero after t=65 (already zero by default)

# Function to calculate the optimal consumption path (perfect consumption smoothing)
def optimal_consumption(e_t, beta, T, r):
    # Total present value of income
    income_present_value = np.sum([e_t[t] / (1 + r)**t for t in range(T+1)])
    
    # Optimal constant consumption level
    x_optimal = income_present_value / np.sum([1 / (1 + r)**t for t in range(T+1)])
    
    # Optimal consumption for all periods (constant)
    return np.ones(T+1) * x_optimal

# Calculate optimal consumption path
x_t_optimal = optimal_consumption(e_t, beta, T, r)

# Calculate savings in each period
s_t = e_t - x_t_optimal

# Calculate wealth accumulation
w_t = np.zeros(T+1)
for t in range(1, T+1):
    w_t[t] = w_t[t-1] * (1 + r) + s_t[t]

# Plot results
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot income and optimal consumption
ax[0].plot(range(T+1), e_t, label="Income ($e_t^1$)", linestyle='--', color='blue')
ax[0].plot(range(T+1), x_t_optimal, label="Optimal Cash Flow ($x_t^1$)", color='green')
ax[0].set_title("Trader's Income and Optimal Cash Flow Over Time")
ax[0].set_xlabel("Time (t)")
ax[0].set_ylabel("Income / Cash Flow")
ax[0].legend()

# Plot savings
ax[1].plot(range(T+1), s_t, label="Savings ($s_t^1$)", color='purple')
ax[1].set_title("Savings Over Time")
ax[1].set_xlabel("Time (t)")
ax[1].set_ylabel("Savings")
ax[1].legend()

# Plot wealth accumulation
ax[2].plot(range(T+1), w_t, label="Wealth ($w_t^1$)", color='orange')
ax[2].set_title("Wealth Accumulation Over Time")
ax[2].set_xlabel("Time (t)")
ax[2].set_ylabel("Wealth")
ax[2].legend()

plt.tight_layout()
plt.show()


# %%
# Adjust income growth for gamma = 3% (growth rate)
gamma = 0.03
e_t_growth = np.zeros(T+1)
e_t_growth[0:35] = 0.5 * (1 + gamma)**np.arange(0, 35)   # Income from t=0 to t=34
e_t_growth[35:65] = 1.0 * (1 + gamma)**np.arange(35, 65)  # Income from t=35 to t=64

# Recalculate optimal consumption path under growth
x_t_optimal_growth = optimal_consumption(e_t_growth, beta, T, r)

# Recalculate savings under growth
s_t_growth = e_t_growth - x_t_optimal_growth

# Recalculate wealth accumulation under growth
w_t_growth = np.zeros(T+1)
for t in range(1, T+1):
    w_t_growth[t] = w_t_growth[t-1] * (1 + r) + s_t_growth[t]

# Plot results under growth scenario
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot income and optimal consumption under growth
ax[0].plot(range(T+1), e_t_growth, label="Income ($e_t^1$) with Growth", linestyle='--', color='blue')
ax[0].plot(range(T+1), x_t_optimal_growth, label="Optimal Cash Flow ($x_t^1$) with Growth", color='green')
ax[0].set_title("Trader's Income and Optimal Cash Flow Over Time (with Growth)")
ax[0].set_xlabel("Time (t)")
ax[0].set_ylabel("Income / Cash Flow")
ax[0].legend()

# Plot savings under growth
ax[1].plot(range(T+1), s_t_growth, label="Savings ($s_t^1$) with Growth", color='purple')
ax[1].set_title("Savings Over Time (with Growth)")
ax[1].set_xlabel("Time (t)")
ax[1].set_ylabel("Savings")
ax[1].legend()

# Plot wealth accumulation under growth
ax[2].plot(range(T+1), w_t_growth, label="Wealth ($w_t^1$) with Growth", color='orange')
ax[2].set_title("Wealth Accumulation Over Time (with Growth)")
ax[2].set_xlabel("Time (t)")
ax[2].set_ylabel("Wealth")
ax[2].legend()

plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters for question b
beta_b = 0.97
gamma_b = 0
theta = 1  # Assuming theta = 1

# Parameters for question c
beta_c = 0.97
gamma_c = 0.1

# Time periods
T = 50  # You can adjust T for a longer or shorter time horizon
t = np.arange(0, T + 1)

# Fisher prices for question b
pt_b = beta_b ** t

# Fisher prices for question c
pt_c = (beta_c * (1 + gamma_c) ** (-theta)) ** t

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(t, pt_b, label='Question b: γ=0, β=0.97', marker='o')
plt.plot(t, pt_c, label='Question c: γ=0.1, β=0.97', marker='s')

plt.title('Time Value of Money (Fisher Prices) Over Time')
plt.xlabel('Time Period (t)')
plt.ylabel('Fisher Price (p̃ₜ)')
plt.legend()
plt.grid(True)
plt.show()



