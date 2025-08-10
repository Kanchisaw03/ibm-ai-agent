import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def _as_dataframe(payload: Dict) -> pd.DataFrame:
    """Flatten selected numeric signals for quick stats/clustering."""
    income_low = float(payload.get("income_distribution", {}).get("low", 0))
    income_mid = float(payload.get("income_distribution", {}).get("mid", 0))
    income_high = float(payload.get("income_distribution", {}).get("high", 0))
    informal = float(payload.get("informal_sector_size", 0))
    population = float(payload.get("population", 0))
    num_industries = float(len(payload.get("key_industries", [])))

    return pd.DataFrame([
        {
            "population": population,
            "income_low": income_low,
            "income_mid": income_mid,
            "income_high": income_high,
            "informal": informal,
            "num_industries": num_industries,
        }
    ])


def analyze_numeric(payload: Dict) -> Dict:
    """Return quick stats and cluster label to help determine zones/risks."""
    df = _as_dataframe(payload)

    # Normalize simple features for KMeans stability when values are zeros
    X = df.values.astype(float)
    if not np.any(X):  # all zeros
        cluster_label = 0
    else:
        try:
            kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
            kmeans.fit(X)
            cluster_label = int(kmeans.labels_[0])
        except Exception as ex:
            logger.warning(f"KMeans failed, falling back to default cluster: {ex}")
            cluster_label = 0

    # Heuristics for predicted risks
    risks: List[str] = []
    if df.loc[0, "informal"] > 0.4:
        risks.append("High dependency on informal sector")
    if df.loc[0, "income_low"] > 0.5:
        risks.append("Income vulnerability in low-income groups")
    if df.loc[0, "population"] > 1_000_000:
        risks.append("Urban congestion and service strain")

    return {
        "cluster": cluster_label,
        "numeric_summary": df.to_dict(orient="records")[0],
        "predicted_risks": risks,
    }


