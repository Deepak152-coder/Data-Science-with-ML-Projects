from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_random_forest_model.pkl")
features = joblib.load(MODEL_DIR / "model_features.pkl")