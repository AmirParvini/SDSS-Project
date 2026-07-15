// ui/ParetoTableView.js
// -----------------------------------------------------------------------------
// The right-side table of Pareto solutions (scenario_id, solution_id, z1..z3).
// Sortable by z1/z2/z3 and reports a row selection through a callback.
//
// SOLID:
//  - SRP: only the DOM of the Pareto panel. It receives plain rows and never
//         reads the raw solution shape or the map.
//  - DIP/Observer: talks to the outside only via onRowSelect (like the other
//         views in this project).
// -----------------------------------------------------------------------------
import {
    PARETO_COLUMNS,
    PARETO_SORT_KEYS,
} from "../config/resultConfig.js";

export default class ParetoTableView {
    constructor(panelSelector = "#paretoPanel") {
        this._panel = $(panelSelector);
        this._thead = this._panel.find("thead");
        this._tbody = this._panel.find("tbody");
        this._rows = [];
        this._sort = { key: null, dir: 1 }; // dir: 1 asc, -1 desc
        this._activeId = null;
        this.onRowSelect = null; // (solutionId) => void
    }

    show() {
        this._panel.removeClass("d-none");
    }

    hide() {
        this._panel.addClass("d-none");
    }

    // rows = [{ scenario_id, solution_id, z1, z2, z3 }]
    render(rows) {
        this._rows = rows.slice();
        this._sort = { key: null, dir: 1 };
        this._activeId = null;
        this._renderHead();
        this._renderBody();
    }

    _renderHead() {
        const cells = PARETO_COLUMNS.map((col) => {
            const sortable = PARETO_SORT_KEYS.includes(col);
            const arrow =
                this._sort.key === col
                    ? this._sort.dir === 1
                        ? " ▲"
                        : " ▼"
                    : sortable
                      ? " ⇅"
                      : "";
            const cls = sortable ? "is-sortable" : "";
            return `<th class="${cls}" data-key="${col}">${col}${arrow}</th>`;
        }).join("");
        this._thead.html(`<tr>${cells}</tr>`);

        this._thead.find("th.is-sortable").on("click", (e) => {
            this._toggleSort($(e.currentTarget).data("key"));
        });
    }

    _toggleSort(key) {
        if (this._sort.key === key) {
            this._sort.dir *= -1;
        } else {
            this._sort = { key, dir: 1 };
        }
        this._sortRows();
        this._renderHead();
        this._renderBody();
    }

    _sortRows() {
        const { key, dir } = this._sort;
        if (!key) return;
        this._rows.sort((a, b) => (Number(a[key]) - Number(b[key])) * dir);
    }

    _renderBody() {
        if (!this._rows.length) {
            this._tbody.html(
                `<tr><td colspan="${PARETO_COLUMNS.length}" class="results-table__empty">No solutions</td></tr>`,
            );
            return;
        }

        const html = this._rows
            .map((row) => {
                const active =
                    row.solution_id == this._activeId ? " is-active" : "";
                const cells = PARETO_COLUMNS.map(
                    (col) => `<td>${this._fmt(row[col])}</td>`,
                ).join("");
                return `<tr class="results-row${active}" data-id="${row.solution_id}">${cells}</tr>`;
            })
            .join("");
        this._tbody.html(html);

        this._tbody.find(".results-row").on("click", (e) => {
            const id = $(e.currentTarget).data("id");
            this.setActive(id);
            if (this.onRowSelect) this.onRowSelect(id);
        });
    }

    setActive(id) {
        this._activeId = id;
        this._tbody.find(".results-row").removeClass("is-active");
        this._tbody
            .find(`.results-row[data-id="${id}"]`)
            .addClass("is-active");
    }

    _fmt(value) {
        if (typeof value === "number") {
            return value.toLocaleString(undefined, {
                maximumFractionDigits: 4,
            });
        }
        return value ?? "-";
    }
}
