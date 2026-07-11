# -*- coding: utf-8 -*-
"""Parses the raw request payload into typed DTOs.

Isolating parsing/validation here keeps RoutingService focused on routing and
makes the accepted input shape explicit and easy to evolve.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .dtos import GeoPoint, RoutingRequestItem


class RequestParser:
    @staticmethod
    def parse_point(raw):
        lng, lat = raw["geometry"]["coordinates"]   # GeoJSON = [lng, lat]
        return GeoPoint(id=raw["id"], lat=float(lat), lng=float(lng))

    @classmethod
    def parse(cls, payload: List[Dict[str, Any]]) -> List[RoutingRequestItem]:
        items: List[RoutingRequestItem] = []
        for entry in payload:
            source = cls.parse_point(entry["source"])
            targets = [cls.parse_point(t) for t in entry.get("targets", [])]
            items.append(RoutingRequestItem(source=source, targets=targets))
        return items
