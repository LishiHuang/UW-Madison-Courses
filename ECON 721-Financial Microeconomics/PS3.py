# %%
import numpy as np
from scipy.optimize import minimize

T = 100
product_range = np.arange(T)

e1 = np.zeros(T)
e1[0:35] = 0.5
e1[35:65] = 1

def e_other(gamma):
    return (1 + gamma) ** product_range

prices = np.ones(T)  

def utility_function(x):
    return np.sum(0.5 ** product_range * np.log(x + 1e-9))

def calculate_demand(prices, holdings):
    def objective(x):
        return -utility_function(x)

    budget = np.sum(prices * holdings)

    cons = {'type': 'eq', 'fun': lambda x: budget - np.sum(prices * x)}
    bounds = [(0, None)] * T  

    result = minimize(objective, holdings, bounds=bounds, constraints=cons, tol=1e-9)
    return result.x

def find_equilibrium(prices, e1, gamma, max_iter=100, alpha=0.1):
    holdings = np.vstack((e1, e_other(gamma) * np.ones((999, T))))  
    for iteration in range(max_iter): 
        demands = calculate_demand(prices, holdings[0]) 
        demand_other = calculate_demand(prices, holdings[1])
        total_demand = demands + 999 * demand_other 
        
        supply = np.sum(holdings, axis=0)  
        
        price_adjustment = total_demand / supply
        prices = prices * (1 - alpha) + prices * alpha * price_adjustment  
        prices = np.clip(prices, 1e-5, None)  #
        
        # print(f"Iteration {iteration}: Prices: {prices}")
    
    return prices

gamma_value = 0.03

equilibrium_prices = find_equilibrium(prices, e1, gamma_value)
print("Equilibrium Prices:", [f"{price:.2f}" for price in equilibrium_prices])

optimal_holdings = calculate_demand(equilibrium_prices, e1)
print("Optimal holdings for trader 1:", [f"{bundle:.2f}" for bundle in optimal_holdings])


# %%
def calculate_demand(prices, holdings):
    def objective(x):
        return -utility_function(x)

    budget = np.sum(prices * holdings)

    cons = {'type': 'eq', 'fun': lambda x: budget - np.sum(prices * x)}
    bounds = [(0, None)] * T  

    initial_guess = np.ones(T) * 2 
    result = minimize(objective, initial_guess, bounds=bounds, constraints=cons, tol=1e-9)
    
    print("Optimization Success:", result.success)
    print("Optimization Status:", result.message)
    
    return result.x

optimal_holdings = calculate_demand(equilibrium_prices, e1)
total_spent = np.sum(equilibrium_prices * optimal_holdings)
print(f"Total spent: {total_spent:.5f}, Budget: {np.sum(equilibrium_prices * e1):.5f}")
print("Optimal holdings for trader 1:", [f"{bundle:.2f}" for bundle in optimal_holdings])


# %%
import matplotlib.pyplot as plt
def plot_holdings(initial_holdings, post_trade_holdings):
    x = np.arange(T)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(x, initial_holdings, label='Initial Holdings', marker='o', linestyle='-', color='blue')
    plt.plot(x, post_trade_holdings, label='Post-Trade Holdings', marker='x', linestyle='-', color='red')
    
    plt.title("Trader 1's Initial and Post-Trade Bundle")
    plt.xlabel('Product Index')
    plt.ylabel('Holdings')
    plt.legend()
    plt.grid()
    plt.ylim(0, np.max(post_trade_holdings) + 1)  # Set y-limit for better visualization
    plt.show()

# 绘制
plot_holdings(e1, optimal_holdings)
# %%

# Create a 2D NumPy array 'A' representing the asset structure of a financial market
# Each column in the matrix corresponds to the promised payment for one of three different commodities
A = np.array([
    [0.1, 0.1, 1.1, 0.1, 1.1],  # Promised quantities for the first commodity across 5 assets
    [0.14, 0.14, 0.2, 1.08, 0.2],  # Promised quantities for the second commodity across 5 assets
    [1.25, 0.3, 0.2, 0.3, 1.2]  # Promised quantities for the third commodity across 5 assets
])

P = np.array([2.065, 0.64, 1.05, 1.58, 2.55])
# %%
def compute_implicit_prices(A, P):
    # Number of commodities (or states)
    num_states = len(A)
    
    # Number of assets in the market
    num_assets = len(P)
    
    # Extract the relevant submatrix of A and corresponding part of P
    subA = A[:, 0:num_states]  # Select the first 'num_states' columns of A
    subP = P[0:num_states]     # Select the first 'num_states' elements of P
    
    # Compute the implicit prices of unitary securities by solving the system
    # SubA * pb = subP
    pb = np.linalg.inv( np.transpose(subA) ) @ subP
    
    # Return the computed implicit prices
    return pb

# Call the function to compute implicit prices with matrix A and price vector P
compute_implicit_prices(A, P)

# %%
def check_arbitrage(A, P, tol = 1e-5):
    # Number of commodities (or states)
    num_states = len(A)
    
    # Number of assets in the market
    num_assets = len(P)
    
    # Compute the implicit prices of unitary securities (basic prices)
    pb = compute_implicit_prices(A, P)
    
    # Iterate over each asset to check for arbitrage opportunities
    for i in range(num_assets):
        # Compute the error for asset i by taking the dot product of the ith column of A with pb
        # and compare it to the price P[i]
        err = abs(np.dot(np.transpose(A)[i], pb) - P[i])
        
        # If the error is larger than a small tolerance (1e-5), arbitrage is detected
        if err > tol:
            print("No arbitrage condition fails for asset", i + 1)
            return True  # Arbitrage opportunity exists
    
    # If no arbitrage opportunities are detected for any asset, return False
    return False

# Example function call to check for arbitrage opportunities in the market
print(check_arbitrage(A, P))  # This will print whether arbitrage exists or not
# %%

def find_arbitrage(A, P):
    # Compute the implicit prices of unitary securities
    pb = compute_implicit_prices(A, P)
    
    # Check if there are any arbitrage opportunities
    if check_arbitrage(A, P) == False:
        print("There are no arbitrage opportunities")
    else:
        print("There are arbitrage opportunities")
        
        # Number of states (commodities) and assets in the market
        num_states = len(A)
        num_assets = len(P)
        
        # Initialize a portfolio of zero holdings for all assets
        portfolio = [0] * len(P)
        
        # Initialize an array to store errors between implied prices and actual prices
        err = [0] * len(P)
        
        # Loop through each asset to find the one with arbitrage opportunity
        for i in range(num_assets):
            # Calculate the error for asset i
            err[i] = np.dot(np.transpose(A)[i], pb) - P[i]
            
            # If the error is significant (greater than tolerance), arbitrage exists for this asset
            if abs(err[i]) > 1e-7:
                # If asset is overpriced, we sell the asset
                if err[i] < 1e-7:
                    portfolio[i] = -1.0  # Sell one unit of asset i
                    assets = np.transpose(A)[i]  # Net assets when selling asset i
                    
                    # Use only the first 'num_states' columns of A
                    subA = A[:, 0:num_states]
                    
                    # Find a portfolio that replicates the sale of 1 unit of asset i
                    subport = np.matmul(np.linalg.inv(subA), assets)
                    
                    # Update portfolio with replicating portfolio
                    portfolio[0:num_states] = subport
                    return portfolio
                
                # If asset is underpriced, we buy the asset
                if err[i] > 1e-7:
                    portfolio[i] = 1.0  # Buy one unit of asset i
                    assets = -np.transpose(A)[i]  # Net assets when buying asset i
                    
                    # Use only the first 'num_states' columns of A
                    subA = A[:, 0:num_states]
                    
                    # Find a portfolio that replicates the purchase of 1 unit of asset i
                    subport = np.matmul(np.linalg.inv(subA), assets)
                    
                    # Update portfolio with replicating portfolio
                    portfolio[0:num_states] = subport
                    return portfolio

# %% 


# Find the arbitrage portfolio for the given prices
portfolio = find_arbitrage(A, P)
print("Arbitrage Portfolio:", portfolio)

    
# %%
# Example scenario where asset 4 is expensive
P_expensive = np.array([2.065, 0.64, 1.05, 2.58, 2.55])

# Find the arbitrage portfolio for overpriced assets
portfolio = find_arbitrage(A, P_expensive)
print("Arbitrage Portfolio:", portfolio)

# Calculate the price of the portfolio (negative because we're profiting from arbitrage)
portfolio_price = -np.dot(P_expensive, portfolio)
print("Price of the Portfolio (Profit from Arbitrage):", portfolio_price)

# Calculate the net acquisition of each commodity (should be 0 for a valid arbitrage)
net_commodity_1 = np.dot(A[0], portfolio)
net_commodity_2 = np.dot(A[1], portfolio)
net_commodity_3 = np.dot(A[2], portfolio)

print("Net Acquisition of Commodity 1:", net_commodity_1)
print("Net Acquisition of Commodity 2:", net_commodity_2)
print("Net Acquisition of Commodity 3:", net_commodity_3)

# Explanation of the arbitrage
if portfolio_price > 0:
    print(f"The agent profits by {portfolio_price} by taking the arbitrage opportunity.")
    print("The agent sells or buys a combination of assets, ensuring that the total net acquisition of each commodity is zero.")
else:
    print("There is no arbitrage opportunity.")
# %%

# Example scenario where asset 4 is cheap
P_cheap = np.array([2.065, 0.64, 1.05, 0.58, 2.55])

# Find the arbitrage portfolio for underpriced assets
portfolio = find_arbitrage(A, P_cheap)
print("Arbitrage Portfolio:", portfolio)

# Calculate the price of the portfolio (negative because we're profiting from arbitrage)
portfolio_price = -np.dot(P_cheap, portfolio)
print("Price of the Portfolio (Profit from Arbitrage):", portfolio_price)

# Calculate the net acquisition of each commodity (should be 0 for a valid arbitrage)
net_commodity_1 = np.dot(A[0], portfolio)
net_commodity_2 = np.dot(A[1], portfolio)
net_commodity_3 = np.dot(A[2], portfolio)

print("Net Acquisition of Commodity 1:", net_commodity_1)
print("Net Acquisition of Commodity 2:", net_commodity_2)
print("Net Acquisition of Commodity 3:", net_commodity_3)

# Explanation of the arbitrage
if portfolio_price > 0:
    print(f"The agent profits by {portfolio_price} by taking the arbitrage opportunity.")
    print("The agent sells or buys a combination of assets, ensuring that the total net acquisition of each commodity is zero.")
else:
    print("There is no arbitrage opportunity.")


# %%

