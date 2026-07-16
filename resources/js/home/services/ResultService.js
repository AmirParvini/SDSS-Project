// services/ResultService.js
// -----------------------------------------------------------------------------
// Data-access layer for the optimization *model*. Its only job is to ask the
// backend to run the model and hand back the raw payload.
//
// SOLID:
//  - SRP: it only talks to the backend for results; no DOM / map logic here.
//  - DIP: the http client is injected (defaults to axios) so it stays testable,
//         exactly like NodeApiService.
//
// The run endpoint URL is injected from Blade (the Run Model button carries it
// as a data attribute) so the exact Laravel route is used, never guessed.
// -----------------------------------------------------------------------------
import axios from "axios";
import { API_BASE_URL } from "../config/nodeConfig.js";

export default class ResultService {
    constructor(httpClient = axios) {
        this.http = httpClient;
    }

    // Runs the model server-side. No body is sent – the controller already has
    // everything it needs. Resolves with the results payload ({ status, data }).
    runModel() {
        return this.http.post(`${API_BASE_URL}/path`).then((res) => res.data);
    }

    fetchReports(scenario_id) {
        return this.http.get(`${API_BASE_URL}/scenario-report/${Number(scenario_id)}`).then((res) => res.data);
    }
}
