import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Loading Dataset...")

df = pd.read_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"
)

df = df.sample(100000, random_state=42)

features = [
    "age",
    "total_price_usd",
    "customer_loyalty_score",
    "total_orders_by_customer"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training KMeans...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["customer_cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts")
print(df["customer_cluster"].value_counts())

df.to_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\customer_segmented.csv",
index=False
)

print("\nCustomer Segmentation Completed")