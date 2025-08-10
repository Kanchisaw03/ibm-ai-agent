import os
import re
import logging
from typing import List, Optional, Tuple

import folium
from folium.plugins import MarkerCluster

logger = logging.getLogger(__name__)

MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_maps")
os.makedirs(MAP_DIR, exist_ok=True)

COORD_REGEX = re.compile(
    r"(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)"
)


def _try_parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    m = COORD_REGEX.search(text or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def generate_livelihood_map(map_id: str, priority_zones: List[str]) -> Optional[str]:
    """Generate a map with markers for zones that look like lat,lon strings.

    Returns the saved HTML path if at least one coordinate is found.
    """
    coordinates: List[Tuple[float, float]] = []
    for zone in priority_zones:
        coords = _try_parse_latlon(zone)
        if coords:
            coordinates.append(coords)

    if not coordinates:
        logger.info("No coordinates found in priority zones; skipping map generation.")
        return None

    # Center on the first coordinate
    m = folium.Map(location=[coordinates[0][0], coordinates[0][1]], zoom_start=12)

    # Add clusters/markers
    cluster = MarkerCluster()
    cluster.add_to(m)
    for lat, lon in coordinates:
        folium.CircleMarker(
            [lat, lon], radius=6, color="#2A9D8F", fill=True, fill_opacity=0.8
        ).add_to(cluster)

    # Draw a simple convex hull-ish bounding box as a "priority area"
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    folium.Rectangle(bounds=bounds, color="#E76F51", fill=False).add_to(m)

    file_path = os.path.join(MAP_DIR, f"{map_id}.html")
    m.save(file_path)
    logger.info(f"Map saved to {file_path}")
    return file_path


