// map/MinimapSync.js
// -----------------------------------------------------------------------------
// رفتارِ همگام‌سازی مینی‌مپ با نقشه‌ی اصلی (کادر قرمزِ محدوده‌ی دید).
// اصل SRP: تنها دلیل تغییر این کلاس، منطق همگام‌سازی مینی‌مپ است.
// مزیت کلیدی: متغیر boundingBox که قبلاً در closureِ whenReady رها بود،
// حالا داخل کلاس کپسوله شده (state پخش‌شده در رویدادها حذف شد).
// -----------------------------------------------------------------------------
export default class MinimapSync {
    constructor(map, minimap, { zoomOffset = 4 } = {}) {
        this.map = map;
        this.minimap = minimap;
        this.zoomOffset = zoomOffset;
        this.boundingBox = null;
    }

    // شروع همگام‌سازی: کادر اولیه + گوش‌دادن به حرکت نقشه‌ی اصلی.
    start() {
        this._drawBox({ opacity: 0.1 });
        this.map.on("move", () => this._onMapMove());
        return this;
    }

    _onMapMove() {
        this._removeBox();
        const center = this.map.getCenter();
        const miniZoom = this.map.getZoom() - this.zoomOffset;
        this._drawBox({ opacity: 0.2 });
        this.minimap.setView(center, miniZoom);
    }

    _drawBox(styleOverrides = {}) {
        const bounds = this.map.getBounds();
        this.boundingBox = L.rectangle(bounds, {
            color: "#ff0000",
            weight: 2,
            fill: true,
            opacity: 0.2,
            ...styleOverrides,
        }).addTo(this.minimap);
    }

    _removeBox() {
        if (this.boundingBox) {
            this.minimap.removeLayer(this.boundingBox);
        }
    }
}
