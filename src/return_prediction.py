import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading Dataset...")

df = pd.read_csv(
r"D:\Nexus Intelligence 2026 Scalable E-Commerce Predictive System\data\featured_ecommerce.csv"
)

df = df.sample(
100000,
random_state=42
)

# Target

df["return_label"] = np.where(
df["order_status"]=="Returned",
1,
0
)

features = [
"rating",
"delivery_days",
"profit_usd",
"customer_loyalty_score",
"quantity"
]

X = df[features]
y = df["return_label"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,
test_size=0.2,
random_state=42
)

model = RandomForestClassifier(
n_estimators=50,
random_state=42,
n_jobs=-1
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

acc = accuracy_score(
y_test,
pred
)

print("\nReturn Prediction Accuracy:")
print(acc)