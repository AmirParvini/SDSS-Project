// config/resultConfig.js
// -----------------------------------------------------------------------------
// Single source of truth for everything related to model *results*.
// OCP: adding/adjusting an allocation type only touches this file; the views,
//      models and controller iterate over these definitions instead of
//      hard-coding names.
// -----------------------------------------------------------------------------

// The four allocation kinds present in every solution object.
export const ALLOCATION_TYPES = [
    "package_flows",
    "shelter_allocations",
    "hospital_allocations",
    "tmc_allocations",
];

// Human friendly labels for the allocation <select> at the bottom table.
export const ALLOCATION_LABELS = {
    package_flows: "Package Flows",
    shelter_allocations: "Shelter Allocations",
    hospital_allocations: "Hospital Allocations",
    tmc_allocations: "TMC Allocations",
};

// For each allocation, which node type is the source and which is the target.
// Used to (a) filter the loaded markers of a solution and (b) resolve a row's
// source_id / target_id to a real marker on the map.
export const ALLOCATION_ENDPOINTS = {
    package_flows: { source: "dc", target: "ec" },
    shelter_allocations: { source: "da", target: "ec" },
    hospital_allocations: { source: "da", target: "h" },
    tmc_allocations: { source: "da", target: "tmc" },
};

// Stroke color used when the geometry of each allocation kind is drawn.
export const ALLOCATION_COLORS = {
    package_flows: "#2563eb", // blue  – goods
    shelter_allocations: "#16a34a", // green – displaced people
    hospital_allocations: "#dc2626", // red   – injured to hospital
    tmc_allocations: "#d97706", // amber – injured to TMC
};

// Objective columns of the Pareto table. z1/z2/z3 map to F1/F2/F3.
export const PARETO_COLUMNS = ["scenario_id", "solution_id", "z1", "z2", "z3"];

// Only these three objective columns are sortable.
export const PARETO_SORT_KEYS = ["z1", "z2", "z3"];

// Order + labels of the cost breakdown shown in the left costs container.
export const COST_FIELDS = [
    { key: "package_flow_cost", label: "Package Flow" },
    { key: "package_cost", label: "Package" },
    { key: "ground_vehicle_cost", label: "Ground Vehicle" },
    { key: "air_vehicle_cost", label: "Air Vehicle" },
    { key: "shelter_establish_cost", label: "Shelter Establish" },
    { key: "tmc_establish_cost", label: "TMC Establish" },
];

// The single field that is the grand total (rendered separately, not in the pie).
export const COST_TOTAL_FIELD = { key: "total_cost", label: "Total Cost" };

// Distinct slice colors for the cost pie chart (index-aligned with COST_FIELDS).
export const COST_COLORS = [
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#d97706",
    "#7c3aed",
    "#0891b2",
];

// Node types that own a capacity shortage, and where to read it from a solution.
export const SHORTAGE_SOURCES = {
    h: [
        { key: "solution_hospital_shortage_severe", label: "Severe Shortage" },
        {
            key: "solution_hospital_shortage_moderate",
            label: "Moderate Shortage",
        },
    ],
    tmc: [{ key: "solution_tmc_shortage", label: "TMC Shortage" }],
};
export const UNSETTLED_POPULATION = {
    da: [{ key: "solution_unsettled_population", label: "Unsettled Population" }]
};
