import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================
# DagsHub + MLflow Setup
# =====================================
dagshub.init(
    repo_owner="Deepak152-coder",
    repo_name="Data-Science-with-ML-Projects",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Deepak152-coder/Data-Science-with-ML-Projects.mlflow"
)

# =====================================
# MLflow Configuration
# =====================================
mlflow.sklearn.autolog(silent=True)

mlflow.set_experiment("bike-demand-hyperparameter-tuning")

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
# Features & Target
# =====================================
X = df.drop("count", axis=1)
y = df["count"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Hyperparameter Combinations
# =====================================
params = [
    {"n_estimators": 50, "max_depth": 10},
    {"n_estimators": 50, "max_depth": 20},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 100, "max_depth": 20},
]

# =====================================
# Hyperparameter Tuning
# =====================================
for param in params:

    with mlflow.start_run(
        run_name=f"RF_{param['n_estimators']}_{param['max_depth']}"
    ):

        model = RandomForestRegressor(
            n_estimators=param["n_estimators"],
            max_depth=param["max_depth"],
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mlflow.log_metric(
            "test_mae",
            mean_absolute_error(y_test, predictions)
        )

        mlflow.log_metric(
            "test_r2",
            r2_score(y_test, predictions)
        )

        print(
            f"Completed -> "
            f"n_estimators={param['n_estimators']}, "
            f"max_depth={param['max_depth']}"
        )

print("\nAll 4 Models Trained Successfully!")