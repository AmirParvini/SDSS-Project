// ui/AllocationTableView.js
// -----------------------------------------------------------------------------
// Full-width bottom table showing the allocations of the selected solution.
// A <select> switches between the four allocation kinds and a search box
// filters across every column. Row clicks are reported through a callback.
//
// SOLID:
//  - SRP: only the DOM of the allocation panel. Column set is derived from the
//         data keys (minus geometry), exactly like the existing NodeTableView.
//  - OCP: allocation kinds / labels / endpoints come from resultConfig.
// -----------------------------------------------------------------------------
import {
    ALLOCATION_TYPES,
    ALLOCATION_LABELS,
    ALLOCATION_ENDPOINTS,
} from "../config/resultConfig.js";

const HIDDEN_COLUMNS = new Set(["geometry"]);

export default class AllocationTableView {
    constructor(panelSelector = "#allocationPanel") {
        this._panel = $(panelSelector);
        this._select = this._panel.find("#allocationSelect");
        this._search = this._panel.find("#allocationSearch");
        this._thead = this._panel.find("thead");
        this._tbody = this._panel.find("tbody");

        this._solution = null;
        this._currentType = ALLOCATION_TYPES[0];
        this.onRowSelect = null; // ({ allocationType, sourceType, sourceId, row }) => void

        this._buildSelect();
        this._wireEvents();
    }

    show() {
        this._panel.removeClass("d-none");
    }

    hide() {
        this._panel.addClass("d-none");
    }

    clear() {
        this._solution = null;
        this._thead.empty();
        this._tbody.empty();
        this._search.val("");
    }

    // Populate the panel from a SolutionModel and render the first kind.
    setSolution(solution) {
        this._solution = solution;
        this._currentType = ALLOCATION_TYPES[0];
        this._select.val(this._currentType);
        this._search.val("");
        this._render();
    }

    _buildSelect() {
        const options = ALLOCATION_TYPES.map(
            (type) =>
                `<option value="${type}">${ALLOCATION_LABELS[type] || type}</option>`,
        ).join("");
        this._select.html(options);
    }

    _wireEvents() {
        this._select.on("change", () => {
            this._currentType = this._select.val();
            this._render();
        });
        this._search.on("input", () => this._renderBody());
    }

    _rows() {
        return this._solution
            ? this._solution.getAllocations(this._currentType)
            : [];
    }

    _columns() {
        const rows = this._rows();
        if (!rows.length) return [];
        return Object.keys(rows[0]).filter((k) => !HIDDEN_COLUMNS.has(k));
    }

    _render() {
        const cols = this._columns();
        this._thead.html(
            `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`,
        );
        this._renderBody();
    }

    _renderBody() {
        const cols = this._columns();
        const term = (this._search.val() || "").trim().toLowerCase();

        const matches = this._rows().filter((row) => {
            if (!term) return true;
            return cols.some((c) =>
                String(row[c] ?? "").toLowerCase().includes(term),
            );
        });

        if (!matches.length) {
            this._tbody.html(
                `<tr><td colspan="${cols.length || 1}" class="results-table__empty">No allocations</td></tr>`,
            );
            return;
        }

        const html = matches
            .map((row) => {
                const cells = cols
                    .map((c) => `<td>${this._fmt(row[c])}</td>`)
                    .join("");
                return `<tr class="results-row" data-source="${row.source_id}">${cells}</tr>`;
            })
            .join("");
        this._tbody.html(html);

        this._tbody.find(".results-row").on("click", (e) => {
            if (!this.onRowSelect) return;
            const sourceId = $(e.currentTarget).data("source");
            const endpoints = ALLOCATION_ENDPOINTS[this._currentType];
            this.onRowSelect({
                allocationType: this._currentType,
                sourceType: endpoints.source,
                sourceId,
            });
        });
    }

    _fmt(value) {
        if (typeof value === "number") {
            return value.toLocaleString(undefined, {
                maximumFractionDigits: 3,
            });
        }
        return value ?? "-";
    }
}
