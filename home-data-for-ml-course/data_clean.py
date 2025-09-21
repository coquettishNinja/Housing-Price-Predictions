import pandas as pd
import numpy as np
from pandas import Series

df = pd.read_csv('../home-data-for-ml-course/train.csv')

print(df.columns)

print(df.head(5))

print(df.describe())

print(df.info())

print(df.shape)

# Show all PoolQC data
print("All PoolQC data:")
print(df['PoolQC'])

print("\n" + "="*50)

# Retrieve only non-NaN PoolQC data
print("Non-NaN PoolQC data:")
non_nan_poolqc = df['PoolQC'].dropna()
print(non_nan_poolqc)

print("\n" + "="*50)

# Show complete rows where PoolQC is not NaN
print("Complete rows with non-NaN PoolQC:")
rows_with_pool = df[df['PoolQC'].notna()]
print(rows_with_pool[['Id', 'PoolArea', 'PoolQC', 'SalePrice']])



