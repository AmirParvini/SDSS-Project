// map/NodeMapView.js
// -----------------------------------------------------------------------------
// همه‌چیزِ مربوط به نقشه: FeatureGroupها، رندر مارکرها، مارکر ضربان‌دار،
// و به‌روزرسانی/حذف نودها روی نقشه.
// اصل SRP: این کلاس فقط «نقشه» را می‌شناسد؛ از DOM، جدول، فرم و API بی‌خبر است.
// ارتباط با بیرون فقط از طریق یک callback (onNodeClick) انجام می‌شود تا
// وابستگی یک‌طرفه بماند (Observer).
// -----------------------------------------------------------------------------
import { NODE_ICONS, PULSING_ICON } from "../config/nodeConfig.js";
import { map } from "./mapInstance.js";

export default class NodeMapView {
    constructor(mapInstance = map) {
        this.map = mapInstance;
        this.featureGroups = this._createFeatureGroups();
        this.pulseMarker = new L.Marker();
        this._onNodeClick = null; // handler ثبت‌شده توسط Controller
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
        this._bindFeatureGroupClicks();
    }

    // ---- رویداد کلیک روی نودها ------------------------------------------------
    onNodeClick(handler) {
        this._onNodeClick = handler;
    }

    _bindFeatureGroupClicks() {
        Object.entries(this.featureGroups).forEach(([type, group]) => {
            group.on("click", (e) => {
                if (!this._onNodeClick) return;
                this._onNodeClick({
                    type,
                    latlng: e.latlng,
                    layer: e.layer,
                    props: e.layer.feature.properties,
                });
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
            if (layer.feature && layer.feature.properties.id == id) {
                Object.keys(newData).forEach((key) => {
                    layer.feature.properties[key] = newData[key];
                });
                if (newData.lat && newData.lng) {
                    layer.setLatLng([newData.lat, newData.lng]);
                    this.showPulse(layer.getLatLng());
                }
                updatedProps = layer.feature.properties;
            }
        });
        return updatedProps;
    }

    // ---- حذف نود از نقشه ------------------------------------------------------
    removeNode(id, type) {
        this.featureGroups[type].eachLayer((layer) => {
            if (layer.feature && layer.feature.properties.id == id) {
                this.featureGroups[type].removeLayer(layer);
                this.map.removeLayer(layer);
            }
        });
        this.removePulse();
    }
}
