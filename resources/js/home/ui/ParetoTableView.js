// ui/ParetoTableView.js
// -----------------------------------------------------------------------------
// The right-side table of Pareto solutions (scenario_id, solution_id, z1..z3).
// Sortable by z1/z2/z3 and reports a row selection through a callback.
// Also owns the inline Topsis toolbar (weight inputs + run button): it reads
// the weights and the rows currently on screen and hands them to the caller.
//
// SOLID:
//  - SRP: only the DOM of the Pareto panel. It receives plain rows and never
//         reads the raw solution shape or the map.
//  - DIP/Observer: talks to the outside only via onRowSelect / onRunTopsis
//         (like the other views in this project); the actual HTTP call is
//         made by the controller through a Topsis service, not here.
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
        this._bestId = null; // solution_id currently highlighted as the Topsis winner
        this.onRowSelect = null; // (solutionId) => void
        this.onRunTopsis = null; // ({ solutionIds, values, weights }) => void

        this._cacheTopsisControls();
        this._wireTopsisControls();
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
        this._bestId = null; // a fresh set of rows invalidates any earlier Topsis result
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

        // Exposes the header's real rendered height as a CSS variable so the
        // pinned best-solution row (see .is-pinned-top in results.css) can
        // stick right below it instead of relying on a hard-coded pixel value.
        this._panel.css("--pareto-thead-h", `${this._thead.outerHeight()}px`);
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

        const html = this._displayRows()
            .map((row) => {
                const active =
                    row.solution_id == this._activeId ? " is-active" : "";
                // Re-applied on every render (including re-sorts) so the
                // highlight survives sorting the table after a Topsis run.
                // is-pinned-top freezes the row in place (see results.css) so
                // the winner stays visible even once the table is scrolled.
                const best =
                    row.solution_id == this._bestId
                        ? " highlight-best-solution is-pinned-top"
                        : "";
                const cells = PARETO_COLUMNS.map(
                    (col) => `<td>${this._fmt(row[col])}</td>`,
                ).join("");
                return `<tr class="results-row${active}${best}" data-id="${row.solution_id}">${cells}</tr>`;
            })
            .join("");
        this._tbody.html(html);

        this._tbody.find(".results-row").on("click", (e) => {
            const id = $(e.currentTarget).data("id");
            this.setActive(id);
            if (this.onRowSelect) this.onRowSelect(id);
        });
    }

    // Rows in the order they should be drawn: the current Topsis winner (if
    // any) always comes first, the rest keep the active sort order. The
    // underlying this._rows/this._sort are untouched, so sorting still works
    // normally on every other row.
    _displayRows() {
        if (this._bestId == null) return this._rows;

        const bestIndex = this._rows.findIndex(
            (row) => row.solution_id == this._bestId,
        );
        if (bestIndex === -1) return this._rows;

        const rows = this._rows.slice();
        const [bestRow] = rows.splice(bestIndex, 1);
        return [bestRow, ...rows];
    }

    setActive(id) {
        this._activeId = id;
        this._tbody.find(".results-row").removeClass("is-active");
        this._tbody
            .find(`.results-row[data-id="${id}"]`)
            .addClass("is-active");
    }

    // Marks `id` as the current Topsis winner: its row is pinned to the top
    // of the table (see _displayRows) and frozen there (.is-pinned-top) so
    // it stays visible no matter how the table is sorted or scrolled.
    highlightBestSolution(id) {
        this._bestId = id;
        this._renderBody();
    }

    // Toggles a busy state on the Topsis button while the request is pending.
    setTopsisLoading(isLoading) {
        this._runTopsisBtn.prop("disabled", isLoading);
    }

    // =========================================================================
    // Topsis toolbar: button + editable weight inputs (z1..z3)
    // =========================================================================
    _cacheTopsisControls() {
        this._runTopsisBtn = this._panel.find("#runTopsisBtn");
        // One input per sortable objective column, e.g. #topsisWeightZ1.
        this._weightInputs = PARETO_SORT_KEYS.map((key) =>
            this._panel.find(`#topsisWeight${key.toUpperCase()}`),
        );
    }

    _wireTopsisControls() {
        this._runTopsisBtn.on("click", () => {
            if (!this._rows.length) {
                alert("There are no Pareto solutions to analyze yet.");
                return;
            }
            if (this.onRunTopsis) {
                this.onRunTopsis({
                    solutionIds: this.getSolutionIds(),
                    values: this.getObjectiveMatrix(),
                    weights: this.getWeights(),
                });
            }
        });
    }

    // The solution ids currently shown in the table, in display order.
    getSolutionIds() {
        return this._rows.map((row) => row.solution_id);
    }

    // Objective matrix aligned with getSolutionIds(), e.g. [[z1, z2, z3], ...].
    getObjectiveMatrix() {
        return this._rows.map((row) =>
            PARETO_SORT_KEYS.map((key) => Number(row[key])),
        );
    }

    // Current user-edited weights, aligned with PARETO_SORT_KEYS (z1, z2, z3).
    getWeights() {
        return this._weightInputs.map((input) => Number(input.val()) || 0);
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
