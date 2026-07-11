# -*- coding: utf-8 -*-
"""Nearest-node lookup, isolated behind NodeLocator."""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .interfaces import CoordinateProjector, NodeLocator


class KDTreeNodeLocator(NodeLocator):
    """Nearest graph node via a KD-tree built in a metric CRS.

    Single Responsibility: given a coordinate, return the closest node id.
    The KD-tree is built once per graph and reused for every query.
    """

    def __init__(self, graph: nx.MultiDiGraph, projector: CoordinateProjector) -> None:
        self._projector = projector
        node_ids = []
        coords = []
        for node_id, data in graph.nodes(data=True):
            node_ids.append(node_id)
            coords.append(self._projector.to_metric(data["x"], data["y"]))
        self._node_ids = np.array(node_ids)
        self._tree = cKDTree(np.array(coords, dtype=float))

    def nearest_node(self, lng: float, lat: float) -> Any:
        x, y = self._projector.to_metric(lng, lat)
        _, idx = self._tree.query([x, y])
        return self._node_ids[idx]
