# -*- coding: utf-8 -*-
"""Route calculation service (SOLID).

Public API:
    from route_service import get_routing_service
    service = get_routing_service()
    output = service.compute_from_payload(input_payload)
"""
from .config import RoutingConfig
from .container import RoutingServiceFactory, get_routing_service
from .dtos import GeoPoint, RouteResult, RoutingRequestItem
from .routing_service import RoutingService

__all__ = [
    "RoutingConfig",
    "RoutingService",
    "RoutingServiceFactory",
    "get_routing_service",
    "GeoPoint",
    "RouteResult",
    "RoutingRequestItem",
]
