import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"
)

print(df.shape)

print(df.describe())

# Chart 1 : Revenue Analysis
top_country = (
df.groupby("country")["total_price_usd"]
.sum()
.sort_values(ascending=False)
.head(10)
)

plt.figure(figsize=(10,5))
top_country.plot(kind="bar")
plt.title("Top Revenue Countries")
plt.show()

# Chart 2 : Profit Analysis
plt.figure(figsize=(10,5))

sns.barplot(
data=df,
x="category",
y="profit_usd"
)

plt.xticks(rotation=45)

plt.show()

# Chart 3 : Customer Segment
plt.figure(figsize=(8,5))

sns.countplot(
data=df,
x="customer_segment"
)

plt.show()

# Chart 4 : Fraud Distribution
plt.figure(figsize=(10,5))

sns.histplot(
df["fraud_risk_score"],
bins=30,
kde=True
)

plt.show()

# Chart 5 : Delivery Status
sns.countplot(
data=df,
x="delivery_status"
)

plt.show()

# Chart 6 : Rating Distribution
sns.histplot(
df["rating"],
bins=10
)

plt.show()

# Chart 7 : Profit Category
sns.countplot(
data=df,
x="profit_category"
)

plt.xticks(rotation=20)

plt.show()

# Chart 8 : Device Analysis
sns.countplot(
data=df,
x="device_type"
)

plt.show()

# Chart 9 : Campaign Source
sns.countplot(
data=df,
x="campaign_source"
)

plt.xticks(rotation=45)

plt.show()

# Chart 10 : Correlation Heatmap
numeric_df = df.select_dtypes(
include=np.number
)

plt.figure(figsize=(15,10))

sns.heatmap(
numeric_df.corr(),
cmap="coolwarm"
)

plt.show()