import pandas as pd

print("Loading Dataset...")

df = pd.read_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"
)

# Top Products by Profit

top_products = (
    df.groupby(
        ["product_name","category","brand"]
    )["profit_usd"]
    .sum()
    .reset_index()
    .sort_values(
        by="profit_usd",
        ascending=False
    )
)

top_products = top_products.head(20)

print("\nTop 20 Recommended Products\n")

print(top_products)

top_products.to_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\recommended_products.csv",
index=False
)

print("\nRecommendation File Saved")