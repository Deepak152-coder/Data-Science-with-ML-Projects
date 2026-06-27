import pandas as pd
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =====================================
# MLflow Experiment
# =====================================
mlflow.set_experiment("bike-demand-tracking")

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("../data/train.csv")

# =====================================
# Feature Engineering
# =====================================
df["datetime"] = pd.to_datetime(df["datetime"])

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

df.drop(columns=["datetime", "casual", "registered"], inplace=True)

# =====================================
# Split Features & Target
# =====================================
X = df.drop("count", axis=1)
y = df["count"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# =====================================
# Model Parameters
# =====================================
params = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}

# =====================================
# MLflow Run
# =====================================
with mlflow.start_run(run_name="random-forest-baseline"):

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # =====================================
    # Actual vs Predicted Figure
    # =====================================
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred, alpha=0.5, color="royalblue")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linewidth=2,
    )

    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    plt.title("Actual vs Predicted")
    plt.grid(True)

    plt.tight_layout()

    # Save the figure
    plt.savefig("actual_vs_predicted.png")

    # Log figure to MLflow
    mlflow.log_artifact("actual_vs_predicted.png")

    plt.close()  

    # Log Parameters
    mlflow.log_params(params)

    # Log Metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log Model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("Model Logged Successfully!")