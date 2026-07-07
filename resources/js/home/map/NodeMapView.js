// map/NodeMapView.js
// -----------------------------------------------------------------------------
// همه‌چیزِ مربوط به نقشه: FeatureGroupها، رندر مارکرها، مارکر ضربان‌دار،
// و به‌روزرسانی/حذف نودها روی نقشه.
// اصل SRP: این کلاس فقط «نقشه» را می‌شناسد؛ از DOM، جدول، فرم و API بی‌خبر است.
// ارتباط با بیرون فقکل از طریق یک callback (onNodeClick) انجام می‌شود تا
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
        this._mapElement = mapInstance.getContainer();
        // چون با تعویض سناریو، renderNodes چندین‌بار اجرا می‌شود، رویدادهای کلیک
        // را فقط «یک‌بار» می‌بندیم تا handlerها تکراری (و چندبار صدا زده) نشوند.
        this._clicksBound = false;
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
                id: node_id
            }
        })
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
}