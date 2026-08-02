// services/TopsisService.js
// -----------------------------------------------------------------------------
// Data-access layer for the Topsis (MCDM) ranking endpoint. Sends the
// solution ids, their objective matrix and the user-defined weights that
// ParetoTableView reads from the table/toolbar, and resolves with the id of
// the best-ranked solution.
//
// SOLID:
//  - SRP: it only talks to the backend for Topsis; no DOM logic here.
//  - DIP: the http client is injected (defaults to axios), exactly like the
//         other services in this project (NodeApiService, ResultService...).
// -----------------------------------------------------------------------------
import axios from "axios";
import { API_BASE_URL } from "../config/nodeConfig.js";

export default class TopsisService {
    constructor(httpClient = axios) {
        this.http = httpClient;
    }

    // solutionIds: [id, ...]
    // values:      [[z1, z2, z3], ...] aligned with solutionIds
    // weights:     [w1, w2, w3] aligned with the values columns
    // Resolves with the best solution id, or rejects with an Error otherwise.
    getBestSolutionId({ solutionIds, values, weights }) {
        return this.http
            .post(`${API_BASE_URL}/api/topsis`, {
                solution_ids: solutionIds,
                values: values,
                weights: weights,
            })
            .then((res) => {
                if (res.data.status !== "success") {
                    throw new Error(res.data.message || "Topsis failed.");
                }
                return res.data.best_solution_id;
            });
    }
}
