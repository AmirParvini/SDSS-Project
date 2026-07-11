# -*- coding: utf-8 -*-
"""Shortest-path planning, isolated behind RoutePlanner."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import osmnx as ox

from .geojson import GeoJsonGeometryBuilder
from .interfaces import RoutePlanner

WEIGHT = "length"  # edge attribute used as the distance metric (metres)


class ShortestPathPlanner(RoutePlanner):
    """Dijkstra shortest path over a NetworkX/OSMnx graph.

    Returns (distance_in_metres, geojson_geometry). Both are None when the
    two nodes are not connected. This class raises no web-framework errors —
    it only speaks the domain language, keeping it reusable and testable.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph
        self._geo = GeoJsonGeometryBuilder()

    def plan(self, source_node: Any, target_node: Any) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        if source_node == target_node:
            data = self._graph.nodes[source_node]
            return 0.0, self._geo.point(data["x"], data["y"])

        try:
            path = ox.shortest_path(self._graph, source_node, target_node, weight=WEIGHT)
        except nx.NetworkXNoPath:
            path = None

        if not path or len(path) < 2:
            return None, None

        distance = float(nx.path_weight(self._graph, path, weight=WEIGHT))
        geometry = self._geo.line_string(self._extract_coords(path))
        return distance, geometry

    def _extract_coords(self, path):
        route_gdf = ox.routing.route_to_gdf(self._graph, path)
        coords = []
        for _, row in route_gdf.iterrows():
            coords.extend(list(row["geometry"].coords))
        return self._geo.dedupe(coords)
