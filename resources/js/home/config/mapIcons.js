// config/mapIcons.js
// -----------------------------------------------------------------------------
// تعریف آیکون‌های Leaflet. اینها «داده/asset» هستند، نه رفتار.
// اصل SRP: تنها دلیل تغییر این فایل، اضافه/حذف/تغییر یک آیکون است.
// -----------------------------------------------------------------------------
export const dc_icon = L.divIcon({
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

export const ec_icon = L.divIcon({
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

export const h_icon = L.icon({
    iconUrl:
        "https://img.icons8.com/?size=100&id=11934&format=png&color=000000",
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    popupAnchor: [0, -5],
});

export const da_icon = L.divIcon({
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

export const tmc_icon = L.icon({
    iconUrl:
        "https://img.icons8.com/?size=100&id=Tc1f4oIX57Up&format=png&color=DE2AB1",
    iconSize: [15, 15],
    iconAnchor: [7.5, 7.5],
    popupAnchor: [0, -7.5],
});

export const puls_icon = L.divIcon({
    className: "custom-pulsing-icon", // کلاس اصلی
    html: `
      <div class="ping-container">
        <div class="ring"></div>
        <div class="ring"></div>
      </div>
    `,
    iconAnchor: [5, 5], // مرکز دایره روی مختصات دقیق
    popupAnchor: [0, -2.5],
});
