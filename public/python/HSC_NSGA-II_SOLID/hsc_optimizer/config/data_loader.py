"""Loading distance data from an external source.

Defines a small port (:class:`DistanceLoader`) so the rest of the system never
knows *where* distances come from.  The default adapter reads the project's
``HSC_Parameters.json``; a test or a different deployment can supply any other
implementation without touching the optimiser.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..domain.problem_data import Distances


class DistanceLoader(ABC):
    """Port: produce the :class:`Distances` for a scenario."""

    @abstractmethod
    def load(self) -> Distances:  # pragma: no cover - interface
        raise NotImplementedError


class JsonDistanceLoader(DistanceLoader):
    """Build :class:`Distances` from the ``HSC_Parameters.json`` layout.

    The original code applied ``// 1000`` (metres -> km, floored) to most paths
    and kept ``da_ec`` distances in metres; that behaviour is preserved here.
    """

    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        self._file_path = file_path
        self._encoding = encoding

    def load(self) -> Distances:
        with open(self._file_path, "r", encoding=self._encoding) as handle:
            response: Dict[str, Any] = json.load(handle)
        return self._build(response)

    @staticmethod
    def _build(response: Dict[str, Any]) -> Distances:
        paths = response["pathes"]

        dc_to_shelter: Dict[str, float] = {}
        da_to_h: Dict[str, float] = {}
        da_to_h_helicopter: Dict[str, float] = {}
        da_to_ec: Dict[str, float] = {}
        da_to_tmc: Dict[str, float] = {}
        da_to_tmc_helicopter: Dict[str, float] = {}

        for row in paths["idc_ec_path"]:
            dc_to_shelter[f"{row['idc_id']},{row['ec_id']}"] = (
                float(row["distance"]) // 1000
            )

        for row in paths["da_h_path"]:
            key = f"{row['da_id']},{row['h_id']}"
            da_to_h[key] = float(row["distance"]) // 1000
            da_to_h_helicopter[key] = float(row["distance_helicopter"]) // 1000

        for row in paths["da_ec_dist"]:
            da_to_ec[f"{row['da_id']},{row['ec_id']}"] = float(row["distance"])

        for row in paths["da_tmc_path"]:
            key = f"{row['da_id']},{row['tmc_id']}"
            da_to_tmc[key] = float(row["distance"]) // 1000
            da_to_tmc_helicopter[key] = float(row["distance_helicopter"]) // 1000

        return Distances(
            dc_to_shelter = dc_to_shelter,
            da_to_ec = da_to_ec,
            da_to_h = da_to_h,
            da_to_h_helicopter = da_to_h_helicopter,
            da_to_tmc = da_to_tmc,
            da_to_tmc_helicopter = da_to_tmc_helicopter,
        )


class InMemoryDistanceLoader(DistanceLoader):
    """Adapter that simply returns a pre-built :class:`Distances` object.

    Useful for tests and for callers that obtain distances elsewhere (e.g. a
    database or routing service)."""

    def __init__(self, distances: Dict) -> None:
        self._distances = distances

    def load(self) -> Distances:

        dc_to_shelter: Dict[str, float] = {}
        da_to_h: Dict[str, float] = {}
        da_to_h_helicopter: Dict[str, float] = {}
        da_to_ec: Dict[str, float] = {}
        da_to_tmc: Dict[str, float] = {}
        da_to_tmc_helicopter: Dict[str, float] = {}

        for row in self._distances["dc_to_ec"]:
            dc_to_shelter[f"{row['source_id']},{row['target_id']}"] = (
                float(row["distance"]) // 1000
            )

        for key, value in self._distances["da_to_h"].items():
            if key == 'ground':
                for ground_dist in value:
                    da_to_h[f"{ground_dist['source_id']},{ground_dist['target_id']}"] = float(ground_dist["distance"]) // 1000
            if key == 'air':
                for air_dist in value:
                    da_to_h_helicopter[f"{air_dist['source_id']},{air_dist['target_id']}"] = float(air_dist["distance"]) // 1000

        for row in self._distances["da_to_ec"]:
            da_to_ec[f"{row['source_id']},{row['target_id']}"] = float(row["distance"])

        for key, value in self._distances["da_to_tmc"].items():
            if key == 'ground':
                for ground_dist in value:
                    da_to_tmc[f"{ground_dist['source_id']},{ground_dist['target_id']}"] = float(ground_dist["distance"]) // 1000
            if key == 'air':
                for air_dist in value:
                    da_to_tmc_helicopter[f"{air_dist['source_id']},{air_dist['target_id']}"] = float(air_dist["distance"]) // 1000

        return Distances(
            dc_to_shelter = dc_to_shelter,
            da_to_ec = da_to_ec,
            da_to_h = da_to_h,
            da_to_h_helicopter = da_to_h_helicopter,
            da_to_tmc = da_to_tmc,
            da_to_tmc_helicopter = da_to_tmc_helicopter,
        )
