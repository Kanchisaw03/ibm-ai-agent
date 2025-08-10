"""
Utility script to create and persist a tiny housing KMeans model to
`housing_agent/housing_risk_model.pkl` for offline runs.

Usage (PowerShell):
  python create_dummy_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "housing_agent")
    out_path = os.path.join(base, "housing_risk_model.pkl")
    os.makedirs(base, exist_ok=True)

    scaler = MinMaxScaler()
    synthetic = pd.DataFrame(
        [
            [20, 500, 70, 20],
            [40, 1500, 50, 40],
            [60, 5000, 30, 60],
            [80, 10000, 15, 80],
            [10, 20000, 85, 10],
        ],
        columns=["condition", "density", "low_income_pct", "access"],
    )
    X = scaler.fit_transform(synthetic.values)
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
    kmeans.fit(X)
    joblib.dump({"scaler": scaler, "kmeans": kmeans}, out_path)
    print(f"Saved dummy model to {out_path}")


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- Robust Path for Saving ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the target directory for the model
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "housing_livelihood_agent")

# Create dummy data
data = {
    'income_level': ['low', 'mid', 'high', 'low', 'mid', 'high', 'low', 'mid', 'high', 'low'],
    'flood_risk_level': ['high', 'moderate', 'low', 'high', 'moderate', 'low', 'high', 'moderate', 'low', 'high'],
    'current_lat': [10.0, 20.0, 30.0, 10.1, 20.1, 30.1, 10.2, 20.2, 30.2, 10.3],
    'current_lon': [100.0, 110.0, 120.0, 100.1, 110.1, 120.1, 100.2, 110.2, 120.2, 100.3],
    'zone_type': [
        'low_risk_commercial', 'residential_near_market', 'high_density_residential',
        'low_risk_commercial', 'residential_near_market', 'high_density_residential',
        'low_risk_commercial', 'residential_near_market', 'high_density_residential',
        'low_risk_commercial'
    ]
}
df = pd.DataFrame(data)

# Preprocess data
le_income = LabelEncoder()
le_flood = LabelEncoder()
le_zone = LabelEncoder()
df['income_level_encoded'] = le_income.fit_transform(df['income_level'])
df['flood_risk_level_encoded'] = le_flood.fit_transform(df['flood_risk_level'])
df['zone_type_encoded'] = le_zone.fit_transform(df['zone_type'])

# Define features and target
features = ['income_level_encoded', 'flood_risk_level_encoded', 'current_lat', 'current_lon']
target = 'zone_type_encoded'
X = df[features]
y = df[target]

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model and encoders to the correct directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_path = os.path.join(OUTPUT_DIR, "relocation_model.pkl")

joblib.dump({
    'model': model,
    'le_income': le_income,
    'le_flood': le_flood,
    'le_zone': le_zone
}, model_path)

print(f"Dummy model and encoders saved to absolute path: {os.path.abspath(model_path)}")
