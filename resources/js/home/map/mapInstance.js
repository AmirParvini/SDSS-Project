// map/mapInstance.js
// -----------------------------------------------------------------------------
// bootstrap زیرساخت نقشه: ساختِ map و minimap، تعریف tileها، ساختِ pane،
// و راه‌اندازی همگام‌سازی مینی‌مپ.
// اصل SRP: مسئولیتش «آماده‌کردن نقشه‌ی پایه» است؛ چیزی درباره‌ی نودها نمی‌داند.
// نکته: رفتار پیچیده‌ی مینی‌مپ به MinimapSync سپرده شده و اینجا فقط compose می‌شود.
// -----------------------------------------------------------------------------
import MinimapSync from "./MinimapSync.js";

export const map = L.map("map", { zoomControl: false }).setView(
    [35.72, 51.4],
    13,
);

export const minimap = L.map("minimap", {
    zoomControl: false,
    scrollWheelZoom: false,
    dragging: false,
    doubleClickZoom: false,
}).setView([35.72, 51.4], 9);

export const map_tiles = {
    1: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    2: "https://tile.openstreetmap.bzh/ca/{z}/{x}/{y}.png",
    3: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    4: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png",
    5: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
};

map.whenReady(() => {
    // pane مخصوص مارکرهای پس‌زمینه (مثل مارکر ضربان‌دار).
    map.createPane("backgroundMarkers");
    map.getPane("backgroundMarkers").style.zIndex = "590";

    // لایه‌ی tile روی نقشه‌ی اصلی و مینی‌مپ.
    L.tileLayer(map_tiles[4]).addTo(map);
    L.tileLayer(map_tiles[4]).addTo(minimap);

    // راه‌اندازی همگام‌سازی مینی‌مپ (رفتار در MinimapSync کپسوله شده).
    new MinimapSync(map, minimap).start();
});
