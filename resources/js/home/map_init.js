export var map = L.map('map').setView([35.72, 51.40], 13);
export const map_tiles = {
    1: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    2: 'https://tile.openstreetmap.bzh/ca/{z}/{x}/{y}.png',
    3: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    4: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png',
}
// L.tileLayer(map_tiles[1], {
//     attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
// }).addTo(map);
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
