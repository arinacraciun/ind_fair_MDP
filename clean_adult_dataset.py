import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("adult.csv")

# Remove rows with "?" in the specified categorical columns
categorical_cols = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex"]
df = df[~df[categorical_cols].apply(lambda x: x.str.contains(r"\?")).any(axis=1)]

# Binary encode "sex" and "income"
binary_cols = ["sex"]
df[binary_cols] = df[binary_cols].apply(lambda x: x.astype('category').cat.codes)

# One-hot encode the remaining categorical variables
one_hot_cols = [col for col in categorical_cols if col not in binary_cols]
df = pd.get_dummies(df, columns=one_hot_cols, dtype=int)

# Columns to normalize
numerical_cols = ["age", "capital.gain", "capital.loss", "hours.per.week"]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Delete not useful columns
del df["fnlwgt"]
del df["education.num"]
del df["native.country"]

# Save to csv
df.to_csv("cleaned_adult.csv")