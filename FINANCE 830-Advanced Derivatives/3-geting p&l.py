import pandas as pd
import numpy as np

# 读取已有的 delta_hedge_positions_updated.xlsx 文件
df = pd.read_excel("delta_hedge_positions.xlsx")
    
for index, row in df.iterrows():
    for rebalance_freq in [1, 2, 4]:
        for path in ["Actual", "Fict. 1", "Fict. 2", "Fict. 3"]:
            name = f"Position_{rebalance_freq}_days_{path}"
            name2 = f"P&L_{rebalance_freq}_days_{path}"
            
            next_index = index + rebalance_freq
            if next_index < len(df):
                next_row_position = df.loc[next_index, name]
                next_row_path_value = df.loc[next_index, path]
                df.at[index, name2] = next_row_position * (row[path] - next_row_path_value)
            else:
                df.at[index, name2] = row[name] * (row[path] - row[path])
df.to_excel("pl.xlsx", index=False)
