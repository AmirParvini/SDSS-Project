// ui/NodeTableView.js
// -----------------------------------------------------------------------------
// جدول «آیتم‌ها»: باز/بسته‌شدن انیمیشنی و پرکردن دینامیک ستون‌ها.
// اصل SRP: فقط DOMِ جدول را می‌شناسد.
// اصل DIP: لایه‌های نقشه را مستقیم نمی‌خواند؛ از طریق layerProvider تزریق‌شده
//         می‌گیرد، و کلیک روی سطر را با یک callback به بیرون می‌دهد.
// -----------------------------------------------------------------------------
export default class NodeTableView {
    constructor() {
        this.currentActiveType = null;
        this._layerProvider = () => []; // (type) => Layer[]
        this._onRowClick = null; // ({ type, id, latlng }) => void
    }

    // منبع لایه‌ها را تزریق می‌کند (معمولاً NodeMapView.getLayers).
    setLayerProvider(fn) {
        this._layerProvider = fn;
    }

    onRowClick(handler) {
        this._onRowClick = handler;
    }

    // اتصال دکمه‌های باز/بسته‌کننده‌ی جدول.
    bindToggle() {
        $(".imgitems").on("click", (e) => {
            const type = $(e.currentTarget).data("type");
            this._toggle(type);
        });
    }

    _toggle(type) {
        const $table = $(".itemstable");

        // هم‌اندازه‌کردن ارتفاع جدول با itemside.
        const itemsideHeight = $(".itemside").innerHeight();
        $table
            .css("max-height", itemsideHeight + "px")
            .css("min-height", itemsideHeight + "px");

        if ($table.hasClass("d-none")) {
            // جدول بسته است → باز کن.
            $table.removeClass("fade-out-left d-none");
            setTimeout(() => $table.addClass("moved"), 10);
            this.currentActiveType = type;
            this.populate(type);
        } else if (this.currentActiveType === type) {
            // همان نوع دوباره کلیک شد → ببند.
            $table.addClass("fade-out-left");
            setTimeout(() => $table.addClass("d-none"), 100);
            $table.removeClass("moved");
            this.currentActiveType = null;
        } else {
            // نوع دیگری انتخاب شد → فقط محتوا را عوض کن.
            this.currentActiveType = type;
            this.populate(type);
        }
    }

    // پرکردن جدول بر اساس لایه‌های نوع انتخاب‌شده.
    populate(type) {
        const table = $(".itemstable table");
        const thead = table.find("thead");
        const tbody = table.find("tbody");

        thead.empty();
        tbody.empty();

        const layers = this._layerProvider(type);
        if (!layers) return;

        if (layers.length === 0) {
            thead.append(`<tr><th scope="col">No Data</th></tr>`);
            tbody.append(
                `<tr><td class="text-center text-muted py-3">No data available</td></tr>`,
            );
            return;
        }

        // اطمینان از وجود lat/lng در پراپرتی‌های همه‌ی لایه‌ها.
        layers.forEach((layer) => {
            const props = this._propsOf(layer);
            const latlng = layer.getLatLng();
            if (props.lat === undefined) props.lat = latlng.lat.toFixed(4);
            if (props.lng === undefined) props.lng = latlng.lng.toFixed(4);
        });

        const keys = Object.keys(this._propsOf(layers[0]));

        // سرستون‌ها
        let headerRow = "<tr>";
        keys.forEach((key) => {
            headerRow += `<th scope="col"><p>${key}</p></th>`;
        });
        headerRow += "</tr>";
        thead.append(headerRow);

        // سطرها
        layers.forEach((layer) => {
            const props = this._propsOf(layer);
            const latlng = layer.getLatLng();

            let rowHtml =
                `<tr style="cursor: pointer;" class="table-row-item"` +
                ` data-id="${props.id || ""}" data-lat="${latlng.lat}" data-lng="${latlng.lng}">`;
            keys.forEach((key) => {
                const val = props[key] !== undefined ? props[key] : "-";
                rowHtml += `<td><p>${val}</p></td>`;
            });
            rowHtml += "</tr>";
            tbody.append(rowHtml);
        });

        this._bindRowClicks(type);
    }

    _bindRowClicks(type) {
        const tbody = $(".itemstable table").find("tbody");
        tbody.find(".table-row-item").on("click", (e) => {
            if (!this._onRowClick) return;
            const $row = $(e.currentTarget);
            this._onRowClick({
                type,
                id: $row.data("id"),
                latlng: {
                    lat: parseFloat($row.data("lat")),
                    lng: parseFloat($row.data("lng")),
                },
            });
        });
    }

    _propsOf(layer) {
        return layer.feature
            ? layer.feature.properties
            : layer.options.properties || {};
    }
}
