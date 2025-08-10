import joblib

model_artifacts = joblib.load("relocation_model.pkl")

model = model_artifacts["model"]
print("✅ Model type:", type(model))
