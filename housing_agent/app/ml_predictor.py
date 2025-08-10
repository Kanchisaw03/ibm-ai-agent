import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .config import settings


logger = logging.getLogger(__name__)


@dataclass
class RiskModel:
    """Container for ML artifacts."""

    scaler: MinMaxScaler
    kmeans: KMeans


_MODEL: RiskModel | None = None


def _model_path() -> str:
    base = settings.MODEL_DIR or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    return os.path.abspath(os.path.join(base, "housing_risk_model.pkl"))


def load_or_init_model() -> None:
    """Load a tiny local model from disk, or initialize a default one.

    The model consists of a `MinMaxScaler` and a `KMeans` with 3 clusters.
    """

    global _MODEL
    path = _model_path()
    if os.path.exists(path):
        try:
            artifacts = joblib.load(path)
            _MODEL = RiskModel(scaler=artifacts["scaler"], kmeans=artifacts["kmeans"])
            logger.info(f"Loaded housing risk model from {path}")
            return
        except Exception as ex:
            logger.warning(f"Failed to load model at {path}: {ex}. Reinitializing.")

    # Initialize a minimal model using synthetic representative data
    scaler = MinMaxScaler()
    # Create a simple grid of plausible values to fit scaler/kmeans
    synthetic = pd.DataFrame(
        [
            # condition, density, low_income%, access
            [20, 500, 70, 20],
            [40, 1500, 50, 40],
            [60, 5000, 30, 60],
            [80, 10000, 15, 80],
            [10, 20000, 85, 10],
        ],
        columns=["condition", "density", "low_income_pct", "access"],
    )
    X = scaler.fit_transform(synthetic.values.astype(float))
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
    kmeans.fit(X)

    _MODEL = RiskModel(scaler=scaler, kmeans=kmeans)
    try:
        joblib.dump({"scaler": scaler, "kmeans": kmeans}, path)
        logger.info(f"Initialized and saved default housing risk model to {path}")
    except Exception as ex:
        logger.warning(f"Could not persist model to {path}: {ex}")


def _vectorize(payload: Dict) -> Tuple[np.ndarray, List[str]]:
    """Convert input into feature vector and candidate risk tags."""

    condition = float(payload.get("housing_condition_score", 0))
    density = float(payload.get("population_density", 0))
    low_income_pct = float(payload.get("low_income_household_percentage", 0))
    access = float(payload.get("access_to_services_score", 0))

    risk_tags: List[str] = []
    if condition < 30:
        risk_tags.append("Housing deterioration")
    if density > 10000:
        risk_tags.append("Overcrowding")
    if low_income_pct > 60:
        risk_tags.append("Poverty concentration")
    if access < 30:
        risk_tags.append("Poor access to services")

    vec = np.array([[condition, density, low_income_pct, access]], dtype=float)
    return vec, risk_tags


def predict_risk_level(payload: Dict) -> Dict:
    """Predict risk cluster and label (low/medium/high) using local KMeans.

    Returns a dict with keys: `risk_level` (str), `cluster` (int),
    `predicted_risks` (List[str]).
    """

    if _MODEL is None:
        load_or_init_model()

    assert _MODEL is not None
    vec, tags = _vectorize(payload)
    scaled = _MODEL.scaler.transform(vec)
    cluster = int(_MODEL.kmeans.predict(scaled)[0])

    # Map cluster index to semantic label deterministically based on centroid
    centroids = _MODEL.kmeans.cluster_centers_
    # Higher density + low access + low condition should correspond to high risk
    scores = np.dot(centroids, np.array([1.0, 1.0, 0.5, -1.0]))  # simple heuristic
    rank = np.argsort(scores)  # ascending
    label_by_cluster = {int(rank[0]): "low", int(rank[1]): "medium", int(rank[2]): "high"}
    risk_level = label_by_cluster.get(cluster, "medium")

    return {
        "risk_level": risk_level,
        "cluster": cluster,
        "predicted_risks": tags,
    }


def find_similar_risk_points(
    payload: Dict,
    dataset_path: Optional[str] = None,
    max_points: int = 3,
) -> List[Tuple[float, float]]:
    """From a small dataset, select points in the same risk cluster.

    Falls back to nearby synthetic offsets if dataset not available.
    """

    try:
        if _MODEL is None:
            load_or_init_model()
        assert _MODEL is not None

        # Compute risk cluster for the input
        vec_in, _ = _vectorize(payload)
        scaled_in = _MODEL.scaler.transform(vec_in)
        cluster_in = int(_MODEL.kmeans.predict(scaled_in)[0])

        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "housing_dataset.csv")
        dataset_path = os.path.abspath(dataset_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)

        df = pd.read_csv(dataset_path)
        # Build feature matrix for dataset
        feats = df[[
            "housing_condition_score",
            "population_density",
            "low_income_household_percentage",
            "access_to_services_score",
        ]].astype(float).values
        scaled = _MODEL.scaler.transform(feats)
        clusters = _MODEL.kmeans.predict(scaled)

        # Select points in same cluster
        same = df[clusters == cluster_in]
        if same.empty:
            raise ValueError("No matching cluster points")

        # Return up to max_points lat/lon
        coords: List[Tuple[float, float]] = list(
            zip(same["latitude"].astype(float).tolist(), same["longitude"].astype(float).tolist())
        )[:max_points]
        return coords
    except Exception as ex:
        logger.info(f"Priority zone dataset selection failed, using offsets: {ex}")
        lat0 = float(payload.get("latitude", 0.0))
        lon0 = float(payload.get("longitude", 0.0))
        return [
            (lat0 + 0.005, lon0 + 0.005),
            (lat0 - 0.004, lon0 + 0.006),
            (lat0 + 0.006, lon0 - 0.004),
        ][:max_points]


