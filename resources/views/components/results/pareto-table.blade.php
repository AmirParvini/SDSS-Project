{{-- Pareto solutions table (right side). Populated by ParetoTableView.js --}}
<div id="paretoPanel" class="results-panel results-panel--right d-none">
    <div class="results-panel__header">
        <span class="results-panel__title">Pareto Solutions</span>
        <span class="results-panel__hint">Click a row to inspect</span>
    </div>
    <div class="results-panel__toolbar">
        {{-- Reads the weight inputs below + the rows currently in the table,
             sends them to the Topsis endpoint and highlights the winning row. --}}
        <button type="button" id="runTopsisBtn" class="results-btn results-btn--primary" title="Rank the solutions below with TOPSIS">
            <i class="fa fa-balance-scale" aria-hidden="true"></i>
            <span class="results-btn__label">TOPSIS</span>
        </button>

        {{-- Editable objective weights (0-1). Read live when the button above is clicked. --}}
        <div class="topsis-weights" id="topsisWeights">
            <div class="topsis-weights__field">
                <label for="topsisWeightZ1">z1</label>
                <input type="number" id="topsisWeightZ1" class="topsis-weights__input" min="0" max="1" step="0.01" value="0.33">
            </div>
            <div class="topsis-weights__field">
                <label for="topsisWeightZ2">z2</label>
                <input type="number" id="topsisWeightZ2" class="topsis-weights__input" min="0" max="1" step="0.01" value="0.33">
            </div>
            <div class="topsis-weights__field">
                <label for="topsisWeightZ3">z3</label>
                <input type="number" id="topsisWeightZ3" class="topsis-weights__input" min="0" max="1" step="0.01" value="0.34">
            </div>
        </div>
    </div>
    <div class="results-panel__body table-wrapper">
        <table class="results-table results-table--pareto">
            <thead></thead>
            <tbody></tbody>
        </table>
    </div>
</div>
