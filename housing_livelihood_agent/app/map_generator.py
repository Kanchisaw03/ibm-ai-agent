import os
import re
import logging
from typing import Optional

import folium

# Configure logging
logger = logging.getLogger(__name__)

# Directory to store generated maps
MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_maps")
os.makedirs(MAP_DIR, exist_ok=True)

# Accept plain "lat,lon" or labeled forms like "Lat 12.34, Lon 56.78"
COORD_REGEX = re.compile(
    r"(?:Lat(?:itude)?[:\s]*)?(-?\d+(?:\.\d+)?)[,\s]+(?:Lon(?:gitude)?[:\s]*)?(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_coords(recommendation: str):
    """Return (lat, lon) tuple if recommendation has comma-separated coordinates."""
    match = COORD_REGEX.search(recommendation)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def generate_map(vendor_id: str, recommendation: str) -> Optional[str]:
    """Generate a Folium map if coordinates are present in recommendation.

    Args:
        vendor_id: identifier used for file naming.
        recommendation: text from LLM; expected lat,lon or area name.

    Returns:
        Path to HTML file or None if coords not found.
    """
    coords = _extract_coords(recommendation)
    if not coords:
        logger.info("No coordinates detected in recommendation – skipping map generation.")
        return None

    lat, lon = coords
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], tooltip="Recommended Zone").add_to(m)

    map_path = os.path.join(MAP_DIR, f"{vendor_id}.html")
    m.save(map_path)
    logger.info(f"Map saved to {map_path}")
    return map_path

