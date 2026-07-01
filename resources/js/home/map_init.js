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

export let dc_icon = L.divIcon({
            className: "",
            html: '<lord-icon\
                src="https://cdn.lordicon.com/jqisugjj.json"\
                trigger="loop"\
                delay="1000"\
                colors="primary:#2516c7"\
                style="width:20px;height:20px">\
                    </lord-icon>',
            iconSize: [20, 20],
            iconAnchor: [10, 10],
            popupAnchor: [0, -5],
        });
export let ec_icon = L.divIcon({
            className: "",
            html: '<lord-icon\
                src="https://cdn.lordicon.com/ewtxwele.json"\
                trigger="loop"\
                delay="1000"\
                colors="primary:#109121"\
                style="width:15px;height:15px">\
                    </lord-icon>',
            iconAnchor: [7.5, 7.5],
            popupAnchor: [0, -7.5],
        });
export let h_icon = L.icon({
            iconUrl:
                "https://img.icons8.com/?size=100&id=11934&format=png&color=000000",
            iconSize: [20, 20],
            iconAnchor: [10, 10],
            popupAnchor: [0, -5],
        });
export let da_icon = L.divIcon({
            className: "",
            html: '<lord-icon\
                src="https://cdn.lordicon.com/izzyzruz.json"\
                trigger="loop"\
                colors="primary:#c71f16"\
                style="width:20px;height:20px;">\
                    </lord-icon>',
            iconSize: [20, 20],
            iconAnchor: [10, 10],
            popupAnchor: [0, -5],
        });
export let tmc_icon = L.icon({
            iconUrl:
                "https://img.icons8.com/?size=100&id=Tc1f4oIX57Up&format=png&color=DE2AB1",
            iconSize: [15, 15],
            iconAnchor: [7.5, 7.5],
            popupAnchor: [0, -7.5],
        });

export let puls_icon = L.divIcon({
        className: "custom-pulsing-icon", // کلاس اصلی
        html: `
      <div class="ping-container">
        <div class="ring" ></div>
        <div class="ring"></div>
      </div>
    `,
        iconAnchor: [5, 5], // قرار دادن مرکز دایره روی مختصات دقیق
        popupAnchor: [0, -2.5],
    }); 
