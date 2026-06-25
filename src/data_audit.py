import pandas as pd

df = pd.read_csv(
    r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\ecommerce_dataset_+1m.csv"
)

missing = df.isnull().sum()

missing = missing[missing > 0]

missing_percent = (missing / len(df)) * 100

audit = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percent": missing_percent
})

audit = audit.sort_values(
    by="Missing_Count",
    ascending=False
)

print(audit)