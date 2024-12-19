import pandas as pd
import numpy as np

df = pd.read_csv('StudentPerformanceFactors.csv')
# print(df.head())
# print(df.info())
# print(df.isnull().sum())

categorical_columns = df.select_dtypes(include=['object']).columns

binary_columns = []
non_binary_columns = []
for column in categorical_columns:
    unique_values = df[column].nunique()

    if unique_values == 2:
        binary_columns.append(column)
    else:
        non_binary_columns.append(column)
binary_columns_encoded = pd.get_dummies(binary_columns,drop_first=False)
print(binary_columns_encoded)