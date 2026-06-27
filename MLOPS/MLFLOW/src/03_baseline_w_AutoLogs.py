import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================================
# Enable MLflow Autologging
# =====================================
mlflow.sklearn.autolog()

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
# Train Model
# =====================================
with mlflow.start_run(run_name="random-forest-autolog"):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Model Logged Successfully!")