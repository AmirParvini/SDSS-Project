// results/SolutionModel.js
// -----------------------------------------------------------------------------
// A thin, read-only wrapper around ONE raw solution object coming from the
// backend. Every "question" the UI asks about a solution is answered here, so
// the views and the controller never touch the raw JSON shape directly.
//
// SOLID:
//  - SRP: it only *queries* a solution; it never renders or fetches anything.
//  - OCP: allocation kinds / endpoints come from resultConfig, so a new kind
//         does not force changes in this class.
// -----------------------------------------------------------------------------
import {
    ALLOCATION_TYPES,
    ALLOCATION_ENDPOINTS,
    SHORTAGE_SOURCES,
} from "../config/resultConfig.js";

export default class SolutionModel {
    constructor(raw) {
        this.raw = raw;
    }

    get id() {
        return this.raw.solution_id;
    }

    // F1/F2/F3 are surfaced to the UI as z1/z2/z3.
    get objectives() {
        return { z1: this.raw.F1, z2: this.raw.F2, z3: this.raw.F3 };
    }

    get costs() {
        return this.raw.costs || {};
    }

    // Allocation rows for a given kind (never geometry-stripped here; callers
    // that render a table decide which columns to hide).
    getAllocations(type) {
        return Array.isArray(this.raw[type]) ? this.raw[type] : [];
    }

    // The set of marker ids (per node type) that belong to this solution.
    // Combines the explicit *_id lists with the source/target ids that appear
    // inside the allocations (this is the only place affected areas surface).
    getNodeIdsByType() {
        const ids = { dc: new Set(), ec: new Set(), h: new Set(), tmc: new Set(), da: new Set() };

        (this.raw.dc_id || []).forEach((id) => ids.dc.add(id));
        (this.raw.ec_id || []).forEach((id) => ids.ec.add(id));
        (this.raw.h_id || []).forEach((id) => ids.h.add(id));
        (this.raw.tmc_id || []).forEach((id) => ids.tmc.add(id));

        ALLOCATION_TYPES.forEach((type) => {
            const { source, target } = ALLOCATION_ENDPOINTS[type];
            this.getAllocations(type).forEach((row) => {
                ids[source].add(row.source_id);
                ids[target].add(row.target_id);
            });
        });

        return ids;
    }

    // Every allocation record in which the given node participates, either as
    // the source or as the target. Each entry keeps its allocation kind so the
    // map can color the geometry and the popup can label the flow.
    getNodeAllocations(nodeType, nodeId) {
        const result = [];
        ALLOCATION_TYPES.forEach((type) => {
            const { source, target } = ALLOCATION_ENDPOINTS[type];
            const asSource = source === nodeType;
            const asTarget = target === nodeType;
            if (!asSource && !asTarget) return;

            this.getAllocations(type).forEach((row) => {
                const isMatch =
                    (asSource && row.source_id == nodeId) ||
                    (asTarget && row.target_id == nodeId);
                if (isMatch) {
                    result.push({
                        type,
                        role: row.source_id == nodeId ? "source" : "target",
                        endpoints: ALLOCATION_ENDPOINTS[type],
                        row,
                    });
                }
            });
        });
        return result;
    }

    // Parsed GeoJSON geometry strings for a node's allocations, ready to draw.
    getNodeGeometries(nodeType, nodeId) {
        return this.getNodeAllocations(nodeType, nodeId)
            .filter((a) => a.row.geometry)
            .map((a) => ({ type: a.type, geometry: this._parse(a.row.geometry) }))
            .filter((g) => g.geometry);
    }

    // Parsed geometry for the allocations that leave a specific source id
    // within a specific allocation kind (used by the bottom-table row click).
    getSourceGeometries(allocationType, sourceId) {
        return this.getAllocations(allocationType)
            .filter((row) => row.source_id == sourceId && row.geometry)
            .map((row) => this._parse(row.geometry))
            .filter(Boolean)
            .map((geometry) => ({ type: allocationType, geometry }));
    }

    // جئومتریِ تخصیص‌هایی که به یک target مشخص می‌رسند (برای تمرکز روی مقصد).
    getTargetGeometries(allocationType, targetId) {
        return this.getAllocations(allocationType)
            .filter((row) => row.target_id == targetId && row.geometry)
            .map((row) => this._parse(row.geometry))
            .filter(Boolean)
            .map((geometry) => ({ type: allocationType, geometry }));
    }

    // جئومتریِ دقیقاً یک تخصیص خاص بین یک source_id و target_id مشخص.
    getAllocationGeometry(allocationType, sourceId, targetId) {
        const row = this.getAllocations(allocationType).find(
            (r) => r.source_id == sourceId && r.target_id == targetId && r.geometry
        );
        if (!row || !row.geometry) return [];
        const geometry = this._parse(row.geometry);
        return geometry ? [{ type: allocationType, geometry }] : [];
    }

    // Capacity shortage(s) for a node, if its type owns any. Returns
    // [{ label, value }] so ec/h/tmc can all be handled uniformly.
    getShortages(nodeType, nodeId) {
        const sources = SHORTAGE_SOURCES[nodeType] || [];
        const out = [];
        sources.forEach(({ key, label }) => {
            const bucket = this.raw[key] || {};
            if (bucket[nodeId] !== undefined) {
                out.push({ label, value: bucket[nodeId] });
            }
        });
        return out;
    }

    _parse(geometryString) {
        try {
            return typeof geometryString === "string"
                ? JSON.parse(geometryString)
                : geometryString;
        } catch (e) {
            console.error("Invalid geometry", e);
            return null;
        }
    }
}
