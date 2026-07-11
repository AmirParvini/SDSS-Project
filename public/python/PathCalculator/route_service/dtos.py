# -*- coding: utf-8 -*-
"""Plain data-transfer objects that define the service's input/output contract.

These carry data only (no behaviour), so every other component agrees on one
stable shape. This is what keeps the public contract independent of the
internal graph/routing implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GeoPoint:
    """A single identified coordinate (a "node" in the request)."""
    id: Any
    lat: float
    lng: float


@dataclass(frozen=True)
class RoutingRequestItem:
    """One source paired with the targets it should be routed to."""
    source: GeoPoint
    targets: List[GeoPoint]


@dataclass(frozen=True)
class RouteResult:
    """One source->target routing outcome.

    `distance` is in metres (None when no path exists).
    `geometry` is a standard GeoJSON *geometry* object (not a Feature),
    e.g. {"type": "LineString", "coordinates": [[lng, lat], ...]} or None.
    """
    source_id: Any
    target_id: Any
    distance: Optional[float]
    geometry: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "distance": self.distance,
            "geometry": self.geometry,
        }
