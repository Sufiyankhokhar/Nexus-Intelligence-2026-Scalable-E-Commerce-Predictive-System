import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading Dataset...")

df = pd.read_csv(r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv")

# Sirf 1 Lakh rows
df = df.sample(n=100000, random_state=42)

print("Rows Used:", len(df))

# Target
df["fraud_flag"] = (df["fraud_risk_score"] > 70).astype(int)

features = [
    "total_price_usd",
    "discount_percent",
    "delivery_days",
    "customer_loyalty_score",
    "quantity",
    "pages_visited",
    "session_duration_minutes",
    "profit_usd"
]

X = df[features]
y = df["fraud_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Model...")

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(classification_report(y_test, pred))

import joblib

joblib.dump(
    model,
    r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\models\fraud_model.pkl"
)

print("Fraud Model Saved")