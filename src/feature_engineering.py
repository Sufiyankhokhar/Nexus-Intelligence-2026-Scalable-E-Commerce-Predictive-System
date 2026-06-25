import pandas as pd
import numpy as np

# Load Clean Dataset
df = pd.read_csv(
    r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\cleaned_ecommerce.csv"
)

# Convert Date Column
df["account_creation_date"] = pd.to_datetime(
    df["account_creation_date"],
    errors="coerce"
)

# FEATURE 1 : Customer Age Group

df["customer_age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 100],
    labels=["Young", "Adult", "Middle_Age", "Senior"]
)

# FEATURE 2 : High Value Customer

median_price = df["total_price_usd"].median()

df["high_value_customer"] = np.where(
    df["total_price_usd"] > median_price,
    1,
    0
)

# FEATURE 3 : Profit Category

df["profit_category"] = pd.cut(
    df["profit_usd"],
    bins=5,
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# FEATURE 4 : Delivery Speed

df["delivery_speed"] = pd.cut(
    df["delivery_days"],
    bins=[0, 3, 7, 14, 100],
    labels=[
        "Fast",
        "Normal",
        "Slow",
        "Very Slow"
    ]
)

# FEATURE 5 : Customer Tenure

df["customer_tenure_days"] = (
    pd.Timestamp.today() -
    df["account_creation_date"]
).dt.days

# FEATURE 6 : Weekend Order Flag

df["weekend_order_flag"] = np.where(
    df["is_weekend"] == "Yes",
    1,
    0
)

# FEATURE 7 : Purchase Power Score

df["purchase_power_score"] = (
    df["customer_loyalty_score"]
    *
    df["total_orders_by_customer"]
)

# SAVE FEATURED DATASET

output_path = r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"

df.to_csv(output_path, index=False)

print("=" * 50)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nNew Features Added:")

print([
    "customer_age_group",
    "high_value_customer",
    "profit_category",
    "delivery_speed",
    "customer_tenure_days",
    "weekend_order_flag",
    "purchase_power_score"
])

print("\nDataset Saved Successfully")