// map/NodeMapView.js
// -----------------------------------------------------------------------------
// همه‌چیزِ مربوط به نقشه: FeatureGroupها، رندر مارکرها، مارکر ضربان‌دار،
// و به‌روزرسانی/حذف نودها روی نقشه.
// اصل SRP: این کلاس فقط «نقشه» را می‌شناسد؛ از DOM، جدول، فرم و API بی‌خبر است.
// ارتباط با بیرون فقکل از طریق یک callback (onNodeClick) انجام می‌شود تا
// وابستگی یک‌طرفه بماند (Observer).
// -----------------------------------------------------------------------------
import { NODE_ICONS, PULSING_ICON } from "../config/nodeConfig.js";
import { ALLOCATION_COLORS } from "../config/resultConfig.js";
import { map } from "./mapInstance.js";

export default class NodeMapView {
    constructor(mapInstance = map) {
        this.map = mapInstance;
        this.featureGroups = this._createFeatureGroups();
        this.pulseMarker = new L.Marker();
        this._onNodeClick = null; // handler ثبت‌شده توسط Controller
        this._mapElement = mapInstance.getContainer();
        // چون با تعویض سناریو، renderNodes چندین‌بار اجرا می‌شود، رویدادهای کلیک
        // را فقط «یک‌بار» می‌بندیم تا handlerها تکراری (و چندبار صدا زده) نشوند.
        this._clicksBound = false;
        // Overlay group that holds the allocation geometries drawn in report mode.
        this._resultOverlay = new L.FeatureGroup();
    }

    // برای هر نوع نقطه یک FeatureGroup جدا می‌سازد.
    _createFeatureGroups() {
        const groups = {};
        Object.keys(NODE_ICONS).forEach((type) => {
            groups[type] = new L.FeatureGroup();
        });
        return groups;
    }

    // ---- رندر داده‌ی اولیه ----------------------------------------------------
    renderNodes(geojson) {
        L.geoJSON(geojson, {
            pointToLayer: (feature, latlng) => {
                const type = feature.properties.type;
                // اصلاح باگ نسخه‌ی قبل: defaultIcon تعریف‌نشده بود.
                const icon = NODE_ICONS[type] ?? new L.Icon.Default();
                const marker = L.marker(latlng, {
                    icon,
                    properties: feature.properties,
                });
                if (this.featureGroups[type]) {
                    marker.addTo(this.featureGroups[type]).addTo(this.map);
                }
                return marker;
            },
        });
        // اتصال رویدادها فقط بار اول؛ FeatureGroupها بین رندرها ثابت می‌مانند.
        if (!this._clicksBound) {
            this._bindFeatureGroupClicks();
            this._bindMapClick();
            this._clicksBound = true;
        }
    }

    // ---- پاک‌سازی کاملِ نقشه (هنگام تعویض/ایجاد سناریو) --------------------------
    // همه‌ی مارکرها را از تمام FeatureGroupها و از روی نقشه حذف می‌کند تا داده‌ی
    // سناریوی جدید از صفر رندر شود. خودِ FeatureGroupها حفظ می‌شوند (بایندِ کلیک باقی می‌ماند).
    clearAll() {
        Object.keys(this.featureGroups).forEach((type) => {
            const group = this.featureGroups[type];
            group.eachLayer((layer) => {
                group.removeLayer(layer);
                this.map.removeLayer(layer);
            });
        });
        this.removePulse();
    }

    // ---- رویداد کلیک روی نودها ------------------------------------------------
    onNodeClick(handler) {
        this._onNodeClick = handler;
    }

    onMapClick(handler) {
        this._onMapClick = handler;
    }

    _bindFeatureGroupClicks() {
        Object.entries(this.featureGroups).forEach(([type, group]) => {
            group.on("click", (e) => {
                if (!this._onNodeClick) return;
                const props = e.layer.feature
                    ? e.layer.feature.properties
                    : e.layer.options.properties || {};
                this._onNodeClick({
                    type,
                    latlng: e.latlng,
                    layer: e.layer,
                    props,
                });
            });
        });
    }

    _bindMapClick() {
        this.map.on("click", (e) => {
            if (!this._onMapClick) return;
            this._onMapClick({
                latlng: e.latlng,
            });
        });
    }

    // ---- مارکر ضربان‌دار (انتخاب فعلی) ---------------------------------------
    showPulse(latlng) {
        this.removePulse();
        this.pulseMarker = L.marker(latlng, {
            icon: PULSING_ICON,
            pane: "backgroundMarkers",
        }).addTo(this.map);
    }

    // ---- تغییر نشانگر موس ----------------------------------------------------
    setCursor(cursorStyle) {
        this._mapElement.style.cursor = cursorStyle;
    }

    resetCursor() {
        this._mapElement.style.cursor = "";
    }

    removePulse() {
        this.pulseMarker.removeFrom(this.map);
    }

    // ---- خواندن لایه‌ها (برای جدول) ------------------------------------------
    getLayers(type) {
        return this.featureGroups[type]
            ? this.featureGroups[type].getLayers()
            : [];
    }

    hasType(type) {
        return Boolean(this.featureGroups[type]);
    }

    // ---- انتخاب از روی جدول --------------------------------------------------
    focus(latlng, zoom = 15) {
        this.map.setView([latlng.lat, latlng.lng], zoom);
    }

    // معادلِ fire('click') در نسخه‌ی قبل: نودِ متناظر با id را روی نقشه
    // «کلیک‌شده» جلوه می‌دهد تا همان مسیر onNodeClick اجرا شود.
    selectLayerById(type, id) {
        const layer = this.getLayers(type).find((l) => {
            const props = l.feature
                ? l.feature.properties
                : l.options.properties || {};
            return props.id == id;
        });
        if (layer) {
            this.featureGroups[type].fire("click", {
                latlng: layer.getLatLng(),
                layer,
            });
        }
    }

    // ---- به‌روزرسانی نود پس از ذخیره -----------------------------------------
    // پراپرتی‌های به‌روزشده را برمی‌گرداند تا Controller بتواند پنل را رفرش کند.
    updateNode(id, type, newData) {
        let updatedProps = null;
        this.featureGroups[type].eachLayer((layer) => {
            const props = layer.feature
                ? layer.feature.properties
                : layer.options.properties || {};
            if (props.id == id) {
                Object.keys(newData).forEach((key) => {
                    if (layer.feature) {
                        layer.feature.properties[key] = newData[key];
                    } else {
                        layer.options.properties[key] = newData[key];
                    }
                });
                if (newData.lat && newData.lng) {
                    layer.setLatLng([newData.lat, newData.lng]);
                    this.showPulse(layer.getLatLng());
                }
                updatedProps = props;
            }
        });
        return updatedProps;
    }

    createNode(node_id, data) {
        let marker = L.marker([data.lat, data.lng], {
            icon: NODE_ICONS[data.type],
            properties: {
                ...data,
                id: node_id,
            },
        });
        marker.addTo(this.featureGroups[data.type]).addTo(map);
        return marker;
    }

    // ---- حذف نود از نقشه ------------------------------------------------------
    removeNode(id, type) {
        this.featureGroups[type].eachLayer((layer) => {
            const props = layer.feature
                ? layer.feature.properties
                : layer.options.properties || {};
            if (props.id == id) {
                this.featureGroups[type].removeLayer(layer);
                this.map.removeLayer(layer);
            }
        });
        this.removePulse();
    }

    // =========================================================================
    // Results / report mode helpers
    // -------------------------------------------------------------------------
    // Everything below is only used while the app is in "report" mode. It keeps
    // all map manipulation inside the map view (SRP): filtering which markers
    // are visible, drawing allocation geometries and opening report popups.
    // =========================================================================

    _propsOf(layer) {
        return layer.feature
            ? layer.feature.properties
            : layer.options.properties || {};
    }

    _findLayer(type, id) {
        return this.getLayers(type).find((l) => this._propsOf(l).id == id);
    }

    // Latitude/longitude of a marker identified by type + id (or null).
    getLatLngById(type, id) {
        const layer = this._findLayer(type, id);
        return layer ? layer.getLatLng() : null;
    }

    getNameById(type, id) {
        const layer = this._findLayer(type, id);
        return layer ? this._propsOf(layer).name : null;
    }

    // Show only the markers that belong to a solution and hide the others.
    // idsByType = { dc:Set, ec:Set, h:Set, tmc:Set, da:Set }
    filterToSolution(idsByType) {
        Object.keys(this.featureGroups).forEach((type) => {
            const allowed = idsByType[type];
            this.getLayers(type).forEach((layer) => {
                const keep = allowed && allowed.has(this._propsOf(layer).id);
                if (keep && !this.map.hasLayer(layer)) {
                    layer.addTo(this.map);
                } else if (!keep && this.map.hasLayer(layer)) {
                    this.map.removeLayer(layer);
                }
            });
        });
    }

    // Bring every marker back onto the map (leaving report mode).
    showAllNodes() {
        Object.keys(this.featureGroups).forEach((type) => {
            this.getLayers(type).forEach((layer) => {
                if (!this.map.hasLayer(layer)) layer.addTo(this.map);
            });
        });
    }

    // Draw the given allocation geometries, colored by their allocation kind.
    // items = [{ type, geometry (parsed GeoJSON) }]
    drawGeometries(items) {
        this.clearGeometries();
        if (!this.map.hasLayer(this._resultOverlay)) {
            this._resultOverlay.addTo(this.map);
        }
        items.forEach(({ type, geometry }) => {
            if (!geometry) return;
            L.geoJSON(geometry, {
                style: {
                    color: ALLOCATION_COLORS[type] || "#334155",
                    weight: 2,
                    opacity: 0.85,
                },
            }).addTo(this._resultOverlay);
        });
    }

    clearGeometries() {
        this._resultOverlay.clearLayers();
    }

    // Bind + open a report popup on the marker (an explicit layer may be passed
    // when the click already produced one).
    openReportPopup(type, id, html, layer = null) {
        const target = layer || this._findLayer(type, id);
        if (!target) return;
        target
            .bindPopup(html, {
                maxWidth: 340,
                className: "result-popup-wrapper",
            })
            .openPopup();
    }

    closePopups() {
        this.map.eachLayer(function (layer) {
            if (layer.getPopup && layer.getPopup()) {
                layer.closePopup();
                layer.unbindPopup();
            }
        });
    }
}
