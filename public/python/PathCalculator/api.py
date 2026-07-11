# -*- coding: utf-8 -*-
"""HTTP entrypoint (recommended for Laravel).

Run as a long-lived microservice so the graph + KD-tree stay in memory and are
built only once. Laravel's controller then calls this over HTTP.

Run:  uvicorn api:app --host 127.0.0.1 --port 8008
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from route_service import get_routing_service

app = FastAPI(title="Route Calculation Service")


class PointModel(BaseModel):
    id: Any
    lat: float
    lng: float


class RequestItemModel(BaseModel):
    source: PointModel
    targets: List[PointModel]


@app.on_event("startup")
def _warm_up() -> None:
    # Build graph/KD-tree once at boot instead of on the first request.
    get_routing_service()


@app.post("/routes")
def compute_routes(payload: List[RequestItemModel]) -> List[Dict[str, Any]]:
    raw = [item.model_dump() for item in payload]
    return get_routing_service().compute_from_payload(raw)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
