# -*- coding: utf-8 -*-
"""The orchestrator that turns the request contract into the result contract.

It depends ONLY on abstractions (GraphProvider, NodeLocator, RoutePlanner)
injected through the constructor — Dependency Inversion. Its single job is to
coordinate them for each source/target pair.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .dtos import GeoPoint, RouteResult, RoutingRequestItem
from .interfaces import NodeLocator, RoutePlanner
from .parser import RequestParser


class RoutingService:
    def __init__(self, node_locator: NodeLocator, route_planner: RoutePlanner) -> None:
        self._node_locator = node_locator
        self._route_planner = route_planner

    def compute(self, items: List[RoutingRequestItem]) -> List[RouteResult]:
        results: List[RouteResult] = []
        for item in items:
            source_node = self._node_locator.nearest_node(item.source.lng, item.source.lat)
            for target in item.targets:
                results.append(self._route_one(item.source, source_node, target))
        return results

    def _route_one(self, source: GeoPoint, source_node: Any, target: GeoPoint) -> RouteResult:
        target_node = self._node_locator.nearest_node(target.lng, target.lat)
        distance, geometry = self._route_planner.plan(source_node, target_node)
        return RouteResult(
            source_id=source.id,
            target_id=target.id,
            distance=distance,
            geometry=geometry,
        )

    # Convenience: accept/emit the raw dict contract directly.
    def compute_from_payload(self, payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = RequestParser.parse(payload)
        return [r.to_dict() for r in self.compute(items)]
