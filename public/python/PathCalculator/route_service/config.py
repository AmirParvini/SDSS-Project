# -*- coding: utf-8 -*-
"""Configuration objects for the routing service.

Grouping configuration into an immutable dataclass keeps every other class
free of environment/IO concerns (Single Responsibility) and makes the whole
system easy to reconfigure without touching business logic (Open/Closed).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RoutingConfig:
    # CRS used for the metric nearest-node search (UTM zone 39N for Tehran).
    target_epsg: int = 32639
    # OSM network type passed to osmnx.
    network_type: str = "drive"
    # Where the downloaded graph is persisted between runs.
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "cache")
    # Name of the persisted graph file inside cache_dir.
    graph_filename: str = "graph.graphml"
    # Ordered list of place queries used to fetch the study-area boundary.
    place_candidates: List[str] = field(
        default_factory=lambda: [
            "District 6, Tehran, Iran",
            "Region 6, Tehran, Iran",
            "Municipal District 6, Tehran, Iran",
            "Tehran 6th District, Iran",
            "منطقه ۶ تهران, تهران, ایران",
        ]
    )
    # osmnx network settings.
    osmnx_timeout: int = 180
    use_osmnx_cache: bool = True

    @property
    def graph_path(self) -> Path:
        return self.cache_dir / self.graph_filename

    @classmethod
    def from_env(cls) -> "RoutingConfig":
        """Build a config from environment variables (used by the entrypoints)."""
        cache_dir = os.environ.get("ROUTING_CACHE_DIR")
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = Path(cache_dir)
        if os.environ.get("TARGET_EPSG"):
            kwargs["target_epsg"] = int(os.environ["TARGET_EPSG"])
        if os.environ.get("NETWORK_TYPE"):
            kwargs["network_type"] = os.environ["NETWORK_TYPE"]
        return cls(**kwargs)
