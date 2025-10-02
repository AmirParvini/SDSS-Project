var map = L.map('map').setView([35.72, 51.40], 13);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);
let lineGroup = L.layerGroup().addTo(map);
let markerGroup = L.layerGroup().addTo(map);
var idc_icon = L.icon({
    iconUrl: 'images/map/distributioncenter.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});
var ec_icon = L.icon({
    iconUrl: 'images/map/shelter.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});
var h_icon = L.icon({
    iconUrl: 'images/map/Hospital.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});
var da_icon = L.divIcon({
    className: "custom-icon",
    html: '<div style="width: 10px; height: 10px; background-color: red; border-radius: 100%;"></div>',
});
var tmc_icon = L.icon({
    iconUrl: 'images/map/Reliefshelter.png',
    iconSize: [20, 20],
    iconAnchor: [10, 20],
    popupAnchor: [0, -20]
});

export{map, lineGroup, markerGroup, idc_icon, da_icon, h_icon, ec_icon, tmc_icon};
