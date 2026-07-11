# -*- coding: utf-8 -*-
"""Coordinate projection concern, isolated behind CoordinateProjector."""
from __future__ import annotations

from typing import Tuple

from pyproj import Transformer

from .interfaces import CoordinateProjector


class PyProjProjector(CoordinateProjector):
    """Projects WGS84 (EPSG:4326) lng/lat to a target metric CRS.

    Single Responsibility: it only knows how to transform coordinates.
    """

    def __init__(self, target_epsg: int) -> None:
        # always_xy=True => inputs/outputs are (lng/x, lat/y) ordered.
        self._transformer = Transformer.from_crs(4326, target_epsg, always_xy=True)

    def to_metric(self, lng: float, lat: float) -> Tuple[float, float]:
        x, y = self._transformer.transform(lng, lat)
        return float(x), float(y)
