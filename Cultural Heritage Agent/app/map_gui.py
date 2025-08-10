from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import folium

from .config import settings


def _marker_color(sentiment: str) -> str:
    s = (sentiment or "").lower()
    if s == "positive":
        return "green"
    if s == "negative":
        return "red"
    return "blue"


def _popup_html(info: Dict[str, str]) -> str:
    citizen = info.get("citizen_id", "")
    sentiment = info.get("sentiment", "")
    grievance = info.get("grievance", "")
    recommendation = info.get("recommendation", "")
    category = info.get("category", "")
    timestamp = info.get("timestamp", "")
    return f"""
    <b>Citizen:</b> {citizen}<br/>
    <b>Timestamp:</b> {timestamp}<br/>
    <b>Sentiment:</b> {sentiment}<br/>
    <b>Category:</b> {category}<br/>
    <b>Grievance:</b> {grievance}<br/>
    <b>Recommendation:</b> {recommendation}
    """


def generate_map(markers: List[Dict]) -> str:
    if not markers:
        # Default map center
        fmap = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    else:
        lat = markers[0]["lat"]
        lon = markers[0]["lon"]
        fmap = folium.Map(location=[lat, lon], zoom_start=12)

    for m in markers:
        color = _marker_color(m.get("sentiment", ""))
        popup = _popup_html(m.get("info", {}))
        folium.Marker(
            location=[m["lat"], m["lon"]],
            popup=popup,
            icon=folium.Icon(color=color),
        ).add_to(fmap)

    settings.maps_dir.mkdir(parents=True, exist_ok=True)
    filename = f"map_{int(time.time()*1000)}.html"
    output_path = settings.maps_dir / filename
    fmap.save(str(output_path))

    # Return URL path that FastAPI will serve via StaticFiles
    return f"/static/maps/{filename}"


