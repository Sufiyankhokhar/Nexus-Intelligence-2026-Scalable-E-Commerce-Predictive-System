import pandas as pd

df = pd.read_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"
)

print("TOTAL REVENUE")
print(df["total_price_usd"].sum())

print()

print("TOTAL PROFIT")
print(df["profit_usd"].sum())

print()

print("TOTAL CUSTOMERS")
print(df["customer_id"].nunique())

print()

print("TOTAL ORDERS")
print(df["order_id"].nunique())

print()

print("AVERAGE ORDER VALUE")
print(df["total_price_usd"].mean())