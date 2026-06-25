import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv(
    r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\ecommerce_dataset_+1m.csv"
)

print("Original Shape:", df.shape)

# DATE CONVERSION

df["order_date"] = pd.to_datetime(df["order_date"])

df["account_creation_date"] = pd.to_datetime(
    df["account_creation_date"],
    dayfirst=True,
    errors="coerce"
)

# MISSING VALUE HANDLING

df["return_reason"] = df["return_reason"].fillna("Not Returned")

df["coupon_code"] = df["coupon_code"].fillna("No Coupon")

df["customer_feedback"] = df["customer_feedback"].fillna("No Feedback")

# CHECK AGAIN

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

# SAVE CLEAN DATA

df.to_csv(
    r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\cleaned_ecommerce.csv",
    index=False
)

print("\nClean Dataset Saved Successfully")
print("Final Shape:", df.shape)