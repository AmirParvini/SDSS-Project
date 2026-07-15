{{-- Cost breakdown container (left side). Populated by CostsReportView.js --}}
<div id="costsPanel" class="results-panel results-panel--left d-none">
    <div class="results-panel__header">
        <span class="results-panel__title" id="costsTitle">Cost Breakdown</span>
    </div>
    <div class="results-panel__body costs-body">
        <div id="costsChart" class="costs-chart"></div>
        <div id="costsLegend" class="costs-legend"></div>
        <div class="costs-total">
            <span>Total Cost</span>
            <b id="costsTotal">0</b>
        </div>
    </div>
</div>