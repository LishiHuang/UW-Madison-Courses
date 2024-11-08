import pandas as pd
import numpy as np

# read delta_values_corrected.xlsx 
df = pd.read_excel("delta_values_corrected.xlsx")

# renew position
for rebalance_freq in [1, 2, 4]:
    for path in ["Actual", "Fict. 1", "Fict. 2", "Fict. 3"]:
        column_name = f"Position_{rebalance_freq}_days_{path}"
        df[column_name] = np.nan

# 设置初始持仓头寸
initial_position = 100

# everyday position
for index, row in df.iterrows():
    for rebalance_freq in [1, 2, 4]:
        if index % rebalance_freq == 0:  # rebalance
            for path in ["Actual", "Fict. 1", "Fict. 2", "Fict. 3"]:
                delta = row[f"Delta {path}"]
                column_name = f"Position_{rebalance_freq}_days_{path}"
                
                # calculate position
                position = round(initial_position * delta)
                
                # updata DataFrame
                df.at[index, column_name] = position
                

# save
df.to_excel("delta_hedge_positions_updated.xlsx", index=False)
