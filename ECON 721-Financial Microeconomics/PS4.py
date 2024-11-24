# %%
import numpy as np

# %%
T = 10
basis = 100
Delta =np.random.randint(3,10, size=(T))
Prices =[basis]
for i in range(T):
    Prices.append(Prices[i] + Delta[i])
Prices = np.array(Prices)
interest_rates = (Prices[1:] - Prices[:-1]) / Prices[:-1]
print([f"{rate:.2%}" for rate in interest_rates])

# %%
Rates  = np.random.rand(T) * 0.2
print([f"{rate:.2%}" for rate in Rates])
Prices = [basis]
for i in range(T):
    Prices.append(Prices[i] * (1 + Rates[i]))
print([f"${price:.2f}" for price in Prices])



