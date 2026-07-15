{{-- Full-width allocations table (bottom). Populated by AllocationTableView.js --}}
<div id="allocationPanel" class="results-panel results-panel--bottom d-none">
    <div class="results-panel__toolbar">
        <span class="results-panel__title">Allocations</span>
        <select id="allocationSelect" class="results-select"></select>
        <input id="allocationSearch" type="search" class="results-search"
            placeholder="Search all columns..." />
    </div>
    <div class="results-panel__body table-wrapper">
        <table class="results-table results-table--allocation">
            <thead></thead>
            <tbody></tbody>
        </table>
    </div>
</div>