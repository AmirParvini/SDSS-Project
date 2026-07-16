{{-- Full-width allocations table (bottom). Populated by AllocationTableView.js --}}
<div id="allocationPanel" class="results-panel results-panel--bottom d-none">
    <div class="results-panel__handle">
        <button id="allocationToggle" type="button" class="results-panel__toggle w-100 d-flex justify-items-center" aria-label="Toggle table">
            <i class="fa-solid fa-chevron-down"></i>
            <i class="fa-solid fa-chevron-up d-none"></i>
        </button>
    </div>
    <div class="results-panel__toolbar">
        <span class="results-panel__title">Allocations</span>
        <select id="allocationSelect" class="results-select"></select>
        <input id="allocationSearch" type="search" class="results-search" placeholder="Search all columns..." />
    </div>
    <div class="results-panel__body table-wrapper">
        <table class="results-table results-table--allocation">
            <thead></thead>
            <tbody></tbody>
        </table>
    </div>
</div>
