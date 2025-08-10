import logging
import os
from typing import List, Optional, Tuple

import folium
from folium.plugins import MarkerCluster


logger = logging.getLogger(__name__)

MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_maps")
os.makedirs(MAP_DIR, exist_ok=True)


def generate_housing_map(
    map_id: str,
    center: Tuple[float, float],
    risk_level: str,
    cluster_points: List[Tuple[float, float]] | None = None,
) -> Optional[str]:
    """Create an interactive map with the location and optional cluster points.

    Args:
        map_id: Unique identifier for saved map file without extension.
        center: (lat, lon) of main location.
        risk_level: "low" | "medium" | "high" to color the marker.
        cluster_points: optional list of (lat, lon) risk zone hints.

    Returns:
        File path to saved HTML, or None if creation failed.
    """

    try:
        color = {"low": "#2A9D8F", "medium": "#F4A261", "high": "#E76F51"}.get(risk_level, "#577590")
        m = folium.Map(location=[center[0], center[1]], zoom_start=13, tiles="cartodbpositron")

        folium.CircleMarker(
            [center[0], center[1]], radius=8, color=color, fill=True, fill_opacity=0.9,
            tooltip=f"Housing risk: {risk_level}"
        ).add_to(m)

        if cluster_points:
            cluster = MarkerCluster()
            cluster.add_to(m)
            for lat, lon in cluster_points:
                folium.CircleMarker([lat, lon], radius=5, color=color, fill=True, fill_opacity=0.7).add_to(cluster)

        file_path = os.path.join(MAP_DIR, f"{map_id}.html")
        m.save(file_path)
        logger.info(f"Map saved to {file_path}")
        return file_path
    except Exception as ex:
        logger.error(f"Failed to generate map: {ex}")
        return None


