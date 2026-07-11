# -*- coding: utf-8 -*-
"""Graph acquisition + caching, isolated behind GraphProvider.

Caching strategy (so the graph is NOT downloaded every time):
  1. In-memory: once loaded, the same instance is reused for the process
     lifetime (kept alive by the composition root / long-running service).
  2. On-disk: a GraphML file in the cache dir. If present it is loaded
     instead of hitting the OSM API.
  3. Only if neither exists is the graph downloaded and then persisted.
"""
from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import osmnx as ox

from .config import RoutingConfig
from .interfaces import GraphProvider

logger = logging.getLogger(__name__)


class OSMGraphProvider(GraphProvider):
    """Provides an OSM drive graph for the configured study area."""

    def __init__(self, config: RoutingConfig) -> None:
        self._config = config
        self._graph: Optional[nx.MultiDiGraph] = None
        self._apply_osmnx_settings()

    def _apply_osmnx_settings(self) -> None:
        ox.settings.use_cache = self._config.use_osmnx_cache
        ox.settings.timeout = self._config.osmnx_timeout
        ox.settings.log_console = False

    def get_graph(self) -> nx.MultiDiGraph:
        if self._graph is not None:                 # (1) in-memory cache
            return self._graph

        self._config.cache_dir.mkdir(parents=True, exist_ok=True)
        graph_path = self._config.graph_path

        if graph_path.exists():                      # (2) on-disk cache
            logger.info("Loading cached graph from %s", graph_path)
            self._graph = ox.load_graphml(graph_path)
            return self._graph

        logger.info("No cached graph found; downloading from OSM")
        self._graph = self._download_graph()         # (3) download + persist
        ox.save_graphml(self._graph, filepath=graph_path)
        logger.info("Graph saved to %s", graph_path)
        return self._graph

    def _download_graph(self) -> nx.MultiDiGraph:
        polygon = self._resolve_boundary()
        return ox.graph_from_polygon(
            polygon,
            network_type=self._config.network_type,
            simplify=True,
            retain_all=False,
        )

    def _resolve_boundary(self):
        for place in self._config.place_candidates:
            try:
                gdf = ox.geocode_to_gdf(place)
                if not gdf.empty:
                    logger.info("Boundary resolved via query: %s", place)
                    return gdf.geometry.union_all()
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                logger.warning("Boundary query failed for '%s': %s", place, exc)
        raise RuntimeError("Could not resolve the study-area boundary from OSM.")
