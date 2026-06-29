"""Entry point for the humanitarian-logistics NSGA-II optimiser.

This is the clean-architecture replacement for the original ``main.py``.  It
only configures and starts the application - all behaviour lives in the
``hsc_optimizer`` package.

Usage::

    python run.py
"""

from __future__ import annotations

import logging
import os

from hsc_optimizer.app import build_optimizer
from hsc_optimizer.config.data_loader import JsonDistanceLoader, InMemoryDistanceLoader
from hsc_optimizer.config.scenario import ScenarioConfig, StdinConfig, StdinData
from hsc_optimizer.optimization.checkpoint import PickleCheckpointRepository
from hsc_optimizer.optimization.factory import ChromosomeFactory
from hsc_optimizer.optimization.initialization import (
    JsonSeededInitializer,
    RandomInitializer,
)
from hsc_optimizer.optimization.nsga2 import NSGA2Config
from hsc_optimizer.optimization.repair import ChromosomeRepair


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    here = os.path.dirname(os.path.abspath(__file__))
    parameters_path = os.path.join(here, "HSC_Parameters.json")

    stdin_data = StdinData.get_data()
    scenario = StdinConfig(data_dict=stdin_data)
    nsga_config = NSGA2Config(
        max_iter=5,
        pop_size=150,
        p_crossover=0.9,
        p_mutation=0.1,
        resume=False,
    )

    distance_loader = InMemoryDistanceLoader(stdin_data['distances'])


    # Optional: seed the initial population from a JSON file, otherwise random.
    seed_path = os.path.join(here, "initial_population.json")
    # The initializer needs a factory; build the same dependencies the app uses.
    from hsc_optimizer.config.scenario import build_problem_data
    problem = build_problem_data(scenario, distance_loader.load())
    factory = ChromosomeFactory(problem, ChromosomeRepair(problem))
    if os.path.exists(seed_path):
        initializer = JsonSeededInitializer(factory, seed_path)
    else:
        initializer = RandomInitializer(factory)

    checkpoint = None

    optimizer = build_optimizer(
        problem,
        nsga_config=nsga_config,
        initializer=initializer,
        checkpoint=checkpoint,
    )

    print(optimizer.solve_to_json())


if __name__ == "__main__":
    main()
