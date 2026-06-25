import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

print("Loading Dataset...")

df = pd.read_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"
)

df = df.sample(100000, random_state=42)

features = [
    "quantity",
    "discount_percent",
    "customer_loyalty_score",
    "delivery_days",
    "pages_visited",
    "session_duration_minutes"
]

target = "total_price_usd"

X = df[features]
y = df[target]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

print("Training Sales Forecast Model...")

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test,pred)
r2 = r2_score(y_test,pred)

print("\nMAE:")
print(mae)

print("\nR2 Score:")
print(r2)