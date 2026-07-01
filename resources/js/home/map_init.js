export const map = L.map('map', {zoomControl: false}).setView([35.72, 51.40], 13);
export const minimap = L.map("minimap", {zoomControl: false, scrollWheelZoom: false, dragging: false, doubleClickZoom: false}).setView([35.72, 51.4], 9);
export const map_tiles = {
    1: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    2: 'https://tile.openstreetmap.bzh/ca/{z}/{x}/{y}.png',
    3: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    4: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png',
    5: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
}
map.whenReady(()=>{
    map.createPane('backgroundMarkers');
    map.getPane('backgroundMarkers').style.zIndex = '590';
    L.tileLayer(map_tiles[4]).addTo(map);
    L.tileLayer(map_tiles[4]).addTo(minimap);
    let boundingBox;
    const bounds = map.getBounds();
    boundingBox = L.rectangle(bounds, { color: "#ff0000", weight: 2, fill: true, opacity: 0.1 }).addTo(minimap);
    map.on("move", function () {
        if (boundingBox) {
            minimap.removeLayer(boundingBox);
        }
        let currentCenter = map.getCenter();
        let currentZoom = map.getZoom();
        let miniZoom = currentZoom - 4;
        const bounds = map.getBounds();
        boundingBox = L.rectangle(bounds, {
            color: "#ff0000",
            weight: 2,
            fill: true,
            opacity: 0.2
        }).addTo(minimap);
        minimap.setView(currentCenter, miniZoom);
    });
});

export let lineGroup = L.layerGroup().addTo(map);
export let markerGroup = L.layerGroup().addTo(map);
export let idc_icon = L.icon({
    iconUrl: 'images/map/distributioncenter.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});
export let ec_icon = L.icon({
    iconUrl: 'images/map/shelter.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});
export let h_icon = L.icon({
    iconUrl: 'images/map/Hospital.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});
export let da_icon = L.divIcon({
    className: "custom-icon",
    html: '<div style="width: 10px; height: 10px; background-color: red; border-radius: 100%;"></div>',
});
export let tmc_icon = L.icon({
    iconUrl: 'images/map/Reliefshelter.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});
