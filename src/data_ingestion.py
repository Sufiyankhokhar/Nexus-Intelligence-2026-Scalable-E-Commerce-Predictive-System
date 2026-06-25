import pandas as pd

df = pd.read_csv(
    r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\ecommerce_dataset_+1m.csv"
)

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n")

print("=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())

print("\n")

print("=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n")

print("=" * 50)
print("TOTAL MISSING VALUES")
print("=" * 50)
print(df.isnull().sum().sum())

print("\n")

print("=" * 50)
print("DUPLICATES")
print("=" * 50)
print(df.duplicated().sum())

print("\n")

print("=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)