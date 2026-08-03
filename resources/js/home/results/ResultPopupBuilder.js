// results/ResultPopupBuilder.js
// -----------------------------------------------------------------------------
// Builds the HTML shown inside a marker popup while in report mode. It explains,
// for the clicked node, which other nodes it is connected to, how much was
// allocated over each connection, and (for ec/h/tmc) any capacity shortage.
//
// SRP: pure presentation. It receives a SolutionModel + the clicked node and
//      returns an HTML string; it never touches the map or the network.
// -----------------------------------------------------------------------------
import { ALLOCATION_LABELS } from "../config/resultConfig.js";

// Friendly labels for the numeric metrics carried by an allocation row.
const METRIC_LABELS = {
    flow: "Flow",
    flow_cost: "Flow Cost",
    g_flow_severe: "Severe · Ground",
    num_gv_severe: "Ground Vehicles (S)",
    g_flow_cost_severe: "Ground Cost (S)",
    a_flow_severe: "Severe · Air",
    num_av_severe: "Air Vehicles (S)",
    a_flow_cost_severe: "Air Cost (S)",
    g_flow_moderate: "Moderate · Ground",
    num_gv_moderate: "Ground Vehicles (M)",
    g_flow_cost_moderate: "Moderate Cost (Ground)",
    a_flow_moderate: "Moderate · Air",
    num_av_moderate: "Air Vehicles (M)",
    a_flow_cost_moderate: "Moderate Cost (Air)",
};

const NON_METRIC_KEYS = new Set(["source_id", "target_id", "geometry"]);

export default class ResultPopupBuilder {
    build(solution, nodeType, nodeId, props = {}) {
        const allocations = solution.getNodeAllocations(nodeType, nodeId);
        const shortages = solution.getShortages(nodeType, nodeId);
        const unsettled_pop = solution.getUnsettledPop(nodeType, nodeId);

        return `
            <div class="result-popup">
                ${this._header(nodeType, nodeId, props)}
                ${this._shortageBlock(shortages)}
                ${this._unsettledBlock(unsettled_pop)}
                ${this._connectionsBlock(allocations)}
            </div>
        `;
    }

    _header(nodeType, nodeId, props) {
        const name = props.name ? this._escape(props.name) : "";
        return `
            <div class="result-popup__head">
                <span class="result-popup__badge result-popup__badge--${nodeType}">${nodeType.toUpperCase()}</span>
                <span class="result-popup__id">#${nodeId}</span>
                ${name ? `<div class="result-popup__name">${name}</div>` : ""}
            </div>
        `;
    }

    _shortageBlock(shortages) {
        if (!shortages.length) return "";
        const items = shortages
            .map(
                (s) =>
                    `<li><span>${s.label}</span><b>${this._num(s.value)}</b></li>`,
            )
            .join("");
        return `
            <div class="result-popup__section result-popup__section--warn">
                <div class="result-popup__title">Capacity Shortage</div>
                <ul class="result-popup__shortage">${items}</ul>
            </div>
        `;
    }

    _unsettledBlock(pops) {
        if (!pops.length) return "";
        const items = pops
            .map(
                (p) =>
                    `<li><span>${p.label}</span><b>${this._num(p.value)}</b></li>`,
            )
            .join("");
        return `
            <div class="result-popup__section result-popup__section--warn">
                <div class="result-popup__title">Unsettled Population</div>
                <ul class="result-popup__unsettled">${items}</ul>
            </div>
        `;
    }

    _connectionsBlock(allocations) {
        if (!allocations.length) {
            return `<div class="result-popup__empty">No connections in this solution.</div>`;
        }

        // Group by allocation kind so each block gets a clear heading.
        const groups = {};
        allocations.forEach((a) => {
            (groups[a.type] = groups[a.type] || []).push(a);
        });

        return Object.entries(groups)
            .map(([type, rows]) => this._connectionGroup(type, rows))
            .join("");
    }

    _connectionGroup(type, rows) {
        const items = rows.map((a) => this._connectionRow(a)).join("");
        return `
            <div class="result-popup__section">
                <div class="result-popup__title">${ALLOCATION_LABELS[type] || type}</div>
                <div class="result-popup__conns">${items}</div>
            </div>
        `;
    }

    _connectionRow(alloc) {
        const { row, role, endpoints } = alloc;
        // The "other" node is the target when the clicked node is the source.
        const otherType = role === "source" ? endpoints.target : endpoints.source;
        const otherId = role === "source" ? row.target_id : row.source_id;
        const arrow = role === "source" ? "→" : "←";

        const metrics = Object.keys(row)
            .filter((k) => !NON_METRIC_KEYS.has(k) && Number(row[k]) > 0)
            .map(
                (k) =>
                    `<div class="result-popup__metric"><span>${METRIC_LABELS[k] || k}</span><b>${this._num(row[k])}</b></div>`,
            )
            .join("");

        return `
            <div class="result-popup__conn">
                <div class="result-popup__conn-head">
                    ${arrow} <span class="result-popup__badge result-popup__badge--${otherType}">${otherType.toUpperCase()} #${otherId}</span>
                </div>
                <div class="result-popup__metrics">${metrics || '<span class="result-popup__muted">no positive flow</span>'}</div>
            </div>
        `;
    }

    _num(value) {
        const n = Number(value);
        if (Number.isNaN(n)) return this._escape(String(value));
        return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
    }

    _escape(text) {
        return String(text).replace(/[&<>"']/g, (c) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
        })[c]);
    }
}
