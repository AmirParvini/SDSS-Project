var map = L.map('map').setView([35.7, 51.39], 11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);
let lineGroup = L.layerGroup().addTo(map);
let markerGroup = L.layerGroup().addTo(map);
var idc_icon = L.icon({
    iconUrl: 'images/map/distributioncenter.png',
    iconSize: [30, 30],
    iconAnchor: [19, 38],
    popupAnchor: [0, -38]
});
var ec_icon = L.icon({
    iconUrl: 'images/map/shelter.png',
    iconSize: [30, 30],
    iconAnchor: [19, 38],
    popupAnchor: [0, -38]
});
var h_icon = L.icon({
    iconUrl: 'images/map/Hospital.png',
    iconSize: [30, 30],
    iconAnchor: [19, 38],
    popupAnchor: [0, -38]
});
var da_icon = L.divIcon({
    className: "custom-icon",
    html: '<div style="width: 6px; height: 6px; background-color: red; border-radius: 100%;"></div>',
});
var tmc_icon = L.icon({
    iconUrl: 'images/map/Reliefshelter.png',
    iconSize: [30, 30],
    iconAnchor: [19, 38],
    popupAnchor: [0, -38]
});

export{map, lineGroup, markerGroup, idc_icon, da_icon, h_icon, ec_icon, tmc_icon};
