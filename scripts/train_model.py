import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

print("Loading dataset...")
df = pd.read_csv("data/features.csv")

# ---- Prepare features ----
X = df.drop(["text", "difficulty"], axis=1)
y = df["difficulty"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training RandomForestRegressor...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Model R² Score: {r2:.4f}")

# ---- Save Model ----
joblib.dump({
    "model": model,
    "feature_names": feature_names
}, "models/difficulty.pkl")

print("Regression model trained and saved successfully.")
