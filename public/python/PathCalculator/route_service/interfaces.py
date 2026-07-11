# -*- coding: utf-8 -*-
"""Abstractions (ports) the service depends on.

Depending on these interfaces instead of concrete classes is the Dependency
Inversion Principle in action: RoutingService never knows whether the graph
came from OSM, a shapefile, or a test stub. Interfaces are kept small
(Interface Segregation) so implementations only implement what they need.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import networkx as nx


class CoordinateProjector(ABC):
    """Projects geodetic (lng, lat) coordinates to a metric CRS."""

    @abstractmethod
    def to_metric(self, lng: float, lat: float) -> Tuple[float, float]:
        ...


class GraphProvider(ABC):
    """Supplies the routable graph, loading from cache when available."""

    @abstractmethod
    def get_graph(self) -> nx.MultiDiGraph:
        ...


class NodeLocator(ABC):
    """Finds the graph node nearest to a geodetic coordinate."""

    @abstractmethod
    def nearest_node(self, lng: float, lat: float) -> Any:
        ...


class RoutePlanner(ABC):
    """Computes distance + GeoJSON geometry between two graph nodes."""

    @abstractmethod
    def plan(self, source_node: Any, target_node: Any) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        ...
