import datetime
from typing import Dict, Any

from .schemas import VendorDataInput

# Constants for risk-based coordinate offsets
RISK_OFFSETS = {
    "high": 0.004,
    "moderate": 0.002,
    "low": 0.0,
}

def compute_safe_zone(lat: float, lon: float, flood_risk: str) -> Dict[str, float]:
    """
    Computes a safer coordinate by applying an offset based on flood risk.
    The offset is added to both latitude and longitude. The resulting
    coordinates are clamped to valid geographical ranges.

    Args:
        lat: The initial latitude.
        lon: The initial longitude.
        flood_risk: The flood risk level ('low', 'moderate', 'high').

    Returns:
        A dictionary containing the new 'latitude' and 'longitude'.
    """
    offset = RISK_OFFSETS.get(flood_risk, 0.0)
    
    new_lat = lat + offset
    new_lon = lon + offset
    
    # Clamp coordinates to their valid ranges to prevent invalid geo data
    new_lat = max(-90.0, min(90.0, new_lat))
    new_lon = max(-180.0, min(180.0, new_lon))
    
    return {"latitude": new_lat, "longitude": new_lon}

def generate_zoning_plan(data: VendorDataInput) -> Dict[str, Any]:
    """
    Generates a full zoning plan dictionary from the input vendor data.
    This includes computing the safe zone and adding a UTC timestamp.

    Args:
        data: A VendorDataInput object containing the vendor's details.

    Returns:
        A dictionary representing the complete zoning plan.
    """
    safe_zone = compute_safe_zone(
        data.current_lat, data.current_lon, data.flood_risk_level
    )
    
    plan = {
        "plan_type": "Vendor_Reallocation",
        "vendor_id": data.vendor_id,
        "new_zone": safe_zone,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
    }
    return plan
