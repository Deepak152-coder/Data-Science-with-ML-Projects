import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# =====================================
# MLflow Experiment
# =====================================
mlflow.set_experiment("nested-runs-Demo_04")

# Enable Auto Logging
mlflow.sklearn.autolog()

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("../data/train.csv")

# Drop datetime column
if "datetime" in df.columns:
    df = df.drop("datetime", axis=1)

# =====================================
# Features & Target
# =====================================
# Remove target leakage columns
X = df.drop(["count", "casual", "registered"], axis=1)
y = df["count"]

# =====================================
# Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Models
# =====================================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

# =====================================
# Parent Run
# =====================================
with mlflow.start_run(run_name="Model Comparison"):

    mlflow.set_tag("Run Type", "Parent")

    print(f"Parent Run ID: {mlflow.active_run().info.run_id}")

    for model_name, model in models.items():

        # =====================================
        # Child Run
        # =====================================
        with mlflow.start_run(
            run_name=model_name,
            nested=True
        ):

            mlflow.set_tag("Run Type", "Child")
            mlflow.set_tag("Model", model_name)

            # Train Model
            model.fit(X_train, y_train)

            print(f"{model_name} Run ID: {mlflow.active_run().info.run_id}")

print("\nAll Nested Runs Logged Successfully!")