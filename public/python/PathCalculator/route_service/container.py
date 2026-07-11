# -*- coding: utf-8 -*-
"""Composition root: wires the concrete implementations together.

This is the only place that knows about concrete classes. It also holds the
process-wide singleton so the (expensive) graph + KD-tree are built ONCE and
reused across all requests.
"""
from __future__ import annotations

from typing import Optional

from .config import RoutingConfig
from .graph_repository import OSMGraphProvider
from .node_locator import KDTreeNodeLocator
from .projection import PyProjProjector
from .route_planner import ShortestPathPlanner
from .routing_service import RoutingService


class RoutingServiceFactory:
    """Builds a fully-wired RoutingService from a config."""

    @staticmethod
    def create(config: Optional[RoutingConfig] = None) -> RoutingService:
        config = config or RoutingConfig.from_env()

        projector = PyProjProjector(config.target_epsg)
        graph = OSMGraphProvider(config).get_graph()   # cached load/download

        node_locator = KDTreeNodeLocator(graph, projector)
        route_planner = ShortestPathPlanner(graph)
        return RoutingService(node_locator, route_planner)


_INSTANCE: Optional[RoutingService] = None


def get_routing_service(config: Optional[RoutingConfig] = None) -> RoutingService:
    """Return the shared RoutingService, building it on first use."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RoutingServiceFactory.create(config)
    return _INSTANCE
