"""Population initialisation strategies.

Encapsulates "how is the first generation built" behind a port so it can vary
independently of the search algorithm (Strategy pattern / OCP): purely random,
or seeded from an external JSON file (e.g. an LLM-generated population) with a
random fallback.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List

from .factory import ChromosomeFactory
from .individual import Individual

logger = logging.getLogger(__name__)


class PopulationInitializer(ABC):
    """Port: build the initial list of (unevaluated) individuals."""

    @abstractmethod
    def initialize(self, pop_size: int) -> List[Individual]:  # pragma: no cover
        raise NotImplementedError


class RandomInitializer(PopulationInitializer):
    """Fill the whole population with random chromosomes."""

    def __init__(self, factory: ChromosomeFactory) -> None:
        self._factory = factory

    def initialize(self, pop_size: int) -> List[Individual]:
        return [Individual(chromosome=self._factory.create_random()) for _ in range(pop_size)]


class JsonSeededInitializer(PopulationInitializer):
    """Seed from a JSON file of chromosomes, then top up with random ones."""

    def __init__(self, factory: ChromosomeFactory, json_path: str) -> None:
        self._factory = factory
        self._json_path = json_path

    def initialize(self, pop_size: int) -> List[Individual]:
        seeded = self._load_seeded(pop_size)
        if not seeded:
            logger.info("No valid seed population found; using random initialisation.")
            return RandomInitializer(self._factory).initialize(pop_size)

        population = [Individual(chromosome=chrom) for chrom in seeded]
        for _ in range(len(seeded), pop_size):
            population.append(Individual(chromosome=self._factory.create_random()))
        logger.info("Seeded %d chromosomes from %s.", len(seeded), self._json_path)
        return population[:pop_size]

    def _load_seeded(self, pop_size: int) -> List:
        if not os.path.exists(self._json_path):
            logger.info("Seed file not found at %s.", self._json_path)
            return []
        try:
            with open(self._json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - defensive: any read/parse error
            logger.warning("Failed to read seed file %s: %s", self._json_path, exc)
            return []

        population = data.get("population") if isinstance(data, dict) else None
        if not population:
            logger.warning("Seed file %s has no usable 'population'.", self._json_path)
            return []

        chromosomes = []
        for item in population:
            if len(chromosomes) >= pop_size:
                break
            chrom = self._factory.from_external_dict(item)
            if chrom is not None:
                chromosomes.append(chrom)
        return chromosomes
