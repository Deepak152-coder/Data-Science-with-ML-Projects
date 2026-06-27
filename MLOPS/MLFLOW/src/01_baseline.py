import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("../data/train.csv")

# ===============================
# Feature Engineering
# ===============================
df["datetime"] = pd.to_datetime(df["datetime"])

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

# Drop unnecessary columns
df.drop(columns=["datetime", "casual", "registered"], inplace=True)

# ===============================
# Split Features & Target
# ===============================
X = df.drop("count", axis=1)
y = df["count"]

# ===============================
# Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ===============================
# Train Random Forest
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Evaluation
# ===============================
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2 Score:", r2_score(y_test, y_pred))

# ===============================
# Save Model
# ===============================
# joblib.dump(model, "bike_demand_rf.pkl")

print("Model Saved Successfully!")