# -*- coding: utf-8 -*-
"""Builds standard GeoJSON *geometry* objects.

Kept separate so the geometry format can change (e.g. add MultiLineString)
without touching routing logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


class GeoJsonGeometryBuilder:
    @staticmethod
    def line_string(coordinates: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        # GeoJSON requires [lng, lat] order and at least two positions.
        return {
            "type": "LineString",
            "coordinates": [[float(lng), float(lat)] for lng, lat in coordinates],
        }

    @staticmethod
    def point(lng: float, lat: float) -> Dict[str, Any]:
        return {"type": "Point", "coordinates": [float(lng), float(lat)]}

    @staticmethod
    def dedupe(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        # Preserve order while removing consecutive/duplicate vertices.
        return list(dict.fromkeys(coords))
