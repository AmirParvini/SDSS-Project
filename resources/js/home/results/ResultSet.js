// results/ResultSet.js
// -----------------------------------------------------------------------------
// Wraps the whole backend payload ({ status, data: [...solutions] }). It hands
// out SolutionModel instances and builds the Pareto table rows.
//
// SRP: it only organizes the collection of solutions; each individual solution
//      is understood by SolutionModel.
// -----------------------------------------------------------------------------
import SolutionModel from "./SolutionModel.js";

export default class ResultSet {
    constructor(payload, scenarioId = null) {
        this.scenarioId = scenarioId;
        this._solutions = (payload && payload.data ? payload.data : []).map(
            (raw) => new SolutionModel(raw),
        );
        this._byId = new Map(this._solutions.map((s) => [String(s.id), s]));
    }

    isEmpty() {
        return this._solutions.length === 0;
    }

    list() {
        return this._solutions;
    }

    getSolution(id) {
        return this._byId.get(String(id)) || null;
    }

    // Rows for the Pareto table: scenario_id, solution_id, z1, z2, z3.
    toParetoRows() {
        return this._solutions.map((s) => {
            const { z1, z2, z3 } = s.objectives;
            return {
                scenario_id: this.scenarioId ?? "-",
                solution_id: s.id,
                z1,
                z2,
                z3,
            };
        });
    }
}
