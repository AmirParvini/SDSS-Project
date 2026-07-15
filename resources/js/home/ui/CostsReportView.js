// ui/CostsReportView.js
// -----------------------------------------------------------------------------
// Left-side container that shows the cost breakdown of the selected solution:
// a graphical pie chart (dependency-free SVG) plus a legend/table and the grand
// total.
//
// SOLID:
//  - SRP: only renders costs. The pie is drawn with plain SVG so no charting
//         library is pulled into the project.
//  - OCP: the cost fields, labels and colors all come from resultConfig.
// -----------------------------------------------------------------------------
import {
    COST_FIELDS,
    COST_TOTAL_FIELD,
    COST_COLORS,
} from "../config/resultConfig.js";

export default class CostsReportView {
    constructor(panelSelector = "#costsPanel") {
        this._panel = $(panelSelector);
        this._chart = this._panel.find("#costsChart");
        this._legend = this._panel.find("#costsLegend");
        this._total = this._panel.find("#costsTotal");
        this._title = this._panel.find("#costsTitle");
    }

    show() {
        this._panel.removeClass("d-none");
    }

    hide() {
        this._panel.addClass("d-none");
    }

    // costs: the solution.costs object. solutionId: for the header.
    render(costs, solutionId) {
        this._title.text(`Cost Breakdown · Solution #${solutionId}`);

        const slices = COST_FIELDS.map((field, i) => ({
            label: field.label,
            value: Number(costs[field.key] || 0),
            color: COST_COLORS[i % COST_COLORS.length],
        })).filter((s) => s.value > 0);

        this._chart.html(this._pieSvg(slices));
        this._legend.html(this._legendHtml(slices));
        this._total.text(this._money(costs[COST_TOTAL_FIELD.key]));
    }

    // Builds an SVG donut/pie from the slices. Uses arc paths on a unit circle.
    _pieSvg(slices) {
        const total = slices.reduce((sum, s) => sum + s.value, 0);
        const size = 180;
        const r = 80;
        const cx = size / 2;
        const cy = size / 2;

        if (total <= 0) {
            return `<svg viewBox="0 0 ${size} ${size}"><circle cx="${cx}" cy="${cy}" r="${r}" fill="#e5e7eb"/></svg>`;
        }

        let angle = -Math.PI / 2; // start at 12 o'clock
        const paths = slices
            .map((s) => {
                const slice = (s.value / total) * Math.PI * 2;
                const x1 = cx + r * Math.cos(angle);
                const y1 = cy + r * Math.sin(angle);
                angle += slice;
                const x2 = cx + r * Math.cos(angle);
                const y2 = cy + r * Math.sin(angle);
                const large = slice > Math.PI ? 1 : 0;
                const d = `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`;
                return `<path d="${d}" fill="${s.color}" stroke="#ffffff" stroke-width="1.5"/>`;
            })
            .join("");

        // Inner white hole turns the pie into a cleaner donut.
        return `
            <svg viewBox="0 0 ${size} ${size}" class="costs-pie">
                ${paths}
                <circle cx="${cx}" cy="${cy}" r="42" fill="var(--results-surface, #ffffff)"/>
            </svg>
        `;
    }

    _legendHtml(slices) {
        const total = slices.reduce((sum, s) => sum + s.value, 0) || 1;
        return slices
            .map((s) => {
                const pct = ((s.value / total) * 100).toFixed(1);
                return `
                    <div class="costs-legend__row">
                        <span class="costs-legend__dot" style="background:${s.color}"></span>
                        <span class="costs-legend__label">${s.label}</span>
                        <span class="costs-legend__pct">${pct}%</span>
                        <span class="costs-legend__value">${this._money(s.value)}</span>
                    </div>
                `;
            })
            .join("");
    }

    _money(value) {
        const n = Number(value || 0);
        return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
}
