"""Composition root.

This module is the *only* place that knows how all the pieces fit together.
Every collaborator is constructed here and injected, so the individual classes
stay free of hidden wiring and remain easy to test or swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config.data_loader import DistanceLoader
from .config.scenario import ScenarioConfig, build_problem_data
from .decoding.decoder import SolutionDecoder
from .decoding.writer import JsonSolutionWriter
from .domain.chromosome import Chromosome
from .domain.problem_data import ProblemData
from .evaluation.allocation import ShelterAllocationService
from .evaluation.assignment import DamageShelterAssigner
from .evaluation.death_probability import DeathProbabilityModel
from .evaluation.objectives import HumanitarianCostEvaluator
from .evaluation.transport import TransportPlanner
from .optimization.checkpoint import CheckpointRepository, NullCheckpointRepository
from .optimization.factory import ChromosomeFactory
from .optimization.initialization import (
    PopulationInitializer,
    RandomInitializer,
)
from .optimization.nsga2 import NSGA2, NSGA2Config
from .optimization.operators import CrossoverOperator, MutationOperator
from .optimization.ranking import DominanceComparator, Ranking
from .optimization.repair import ChromosomeRepair
from .optimization.selection import CrowdingTournamentSelection

import json

@dataclass
class HumanitarianOptimizer:
    """High-level facade: optimise and decode in one call."""

    problem: ProblemData
    engine: NSGA2
    decoder: SolutionDecoder
    factory: ChromosomeFactory
    writer: JsonSolutionWriter

    def optimize(self) -> List[Chromosome]:
        """Run NSGA-II and return the Pareto-front chromosomes."""
        return self.engine.run()

    def solve(self) -> List[dict]:
        """Run the optimiser and decode the front into solution records."""
        return self.decoder.decode(self.optimize())

    def solve_to_json(self, path: Optional[str] = None) -> str:
        """Solve and serialise; optionally also write to ``path``."""
        solutions = self.solve()
        if path:
            self.writer.to_file(solutions, path)
        return json.dumps(solutions, ensure_ascii=False, indent=2)


def build_optimizer(
    problem: ProblemData,    
    nsga_config: Optional[NSGA2Config] = None,
    *,
    initializer: Optional[PopulationInitializer] = None,
    checkpoint: Optional[CheckpointRepository] = None,
) -> HumanitarianOptimizer:
    """Assemble a fully-wired :class:`HumanitarianOptimizer`."""
    nsga_config = nsga_config or NSGA2Config()

    # Shared services (used by both the evaluator and the decoder -> DRY).
    repair = ChromosomeRepair(problem)
    factory = ChromosomeFactory(problem, repair)
    assigner = DamageShelterAssigner(problem)
    allocator = ShelterAllocationService(problem)
    transport = TransportPlanner(problem)
    death_model = DeathProbabilityModel()

    evaluator = HumanitarianCostEvaluator(problem, assigner, allocator, transport, death_model)

    engine = NSGA2(
        evaluator=evaluator,
        initializer=initializer or RandomInitializer(factory),
        crossover=CrossoverOperator(problem, repair),
        mutation=MutationOperator(problem, repair),
        selection=CrowdingTournamentSelection(),
        ranking=Ranking(DominanceComparator()),
        config=nsga_config,
        checkpoint=checkpoint or NullCheckpointRepository(),
    )

    decoder = SolutionDecoder(problem, assigner, allocator, transport)

    return HumanitarianOptimizer(
        problem=problem,
        engine=engine,
        decoder=decoder,
        factory=factory,
        writer=JsonSolutionWriter(),
    )
