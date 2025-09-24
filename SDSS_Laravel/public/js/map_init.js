var map = L.map('map').setView([35.7, 51.39], 11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);
let lineGroup = L.layerGroup().addTo(map);
let markerGroup = L.layerGroup().addTo(map);
var blueCircle = L.divIcon({
    className: 'custom-icon',
    html: '<div style="width: 10px; height: 10px; background-color: blue; border-radius: 50%;"></div>',
});
var redSquare = L.divIcon({
    className: 'custom-icon',
    html: '<div style="width: 6px; height: 6px; background-color: red; border-radius: 50%;"></div>',
});

export{map, lineGroup, markerGroup, blueCircle, redSquare};
