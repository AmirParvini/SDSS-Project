import axios from "axios";
import {
    da_icon,
    dc_icon,
    ec_icon,
    h_icon,
    map,
    puls_icon,
    tmc_icon,
} from "./map_init.js";

$(function () {
    let currentActiveType = null;

    // fade-in/out itmes table
    $(".imgitems").on("click", function (e) {
        const type = $(this).data("type");

        // Dynamically adjust max-height of table to match itemside height
        const itemsideHeight = $(".itemside").innerHeight();
        $(".itemstable")
            .css("max-height", itemsideHeight + "px")
            .css("min-height", itemsideHeight + "px");

        if ($(".itemstable").hasClass("d-none")) {
            $(".itemstable").removeClass("fade-out-left d-none");
            setTimeout(() => {
                $(".itemstable").addClass("moved");
            }, 10);
            currentActiveType = type;
            populateTable(type);
        } else {
            if (currentActiveType === type) {
                $(".itemstable").addClass("fade-out-left");
                setTimeout(() => {
                    $(".itemstable").addClass("d-none");
                }, 100);
                $(".itemstable").removeClass("moved");
                currentActiveType = null;
            } else {
                currentActiveType = type;
                populateTable(type);
            }
        }
    });

    function populateTable(type) {
        const table = $(".itemstable table");
        const thead = table.find("thead");
        const tbody = table.find("tbody");

        thead.empty();
        tbody.empty();

        if (!featureGroups[type]) return;

        const layers = featureGroups[type].getLayers();
        if (layers.length === 0) {
            thead.append(`<tr><th scope="col">No Data</th></tr>`);
            tbody.append(
                `<tr><td class="text-center text-muted py-3">No data available</td></tr>`,
            );
            return;
        }

        // Ensure lat and lng are present in all layer properties dynamically
        layers.forEach((layer) => {
            const props = layer.feature
                ? layer.feature.properties
                : layer.options.properties || {};
            const latlng = layer.getLatLng();
            if (props.lat === undefined) {
                props.lat = latlng.lat.toFixed(4);
            }
            if (props.lng === undefined) {
                props.lng = latlng.lng.toFixed(4);
            }
        });

        const firstLayerProps = layers[0].feature
            ? layers[0].feature.properties
            : layers[0].options.properties || {};
        const keys = Object.keys(firstLayerProps);

        let headerRow = "<tr>";
        keys.forEach((key) => {
            headerRow += `<th scope="col"><p>${key}</p></th>`;
        });
        headerRow += "</tr>";
        thead.append(headerRow);

        layers.forEach((layer) => {
            const props = layer.feature
                ? layer.feature.properties
                : layer.options.properties || {};
            const latlng = layer.getLatLng();

            let rowHtml = `<tr style="cursor: pointer;" class="table-row-item" data-id="${props.id || ""}" data-lat="${latlng.lat}" data-lng="${latlng.lng}">`;
            keys.forEach((key) => {
                const val = props[key] !== undefined ? props[key] : "-";
                rowHtml += `<td><p>${val}</p></td>`;
            });
            rowHtml += "</tr>";
            tbody.append(rowHtml);
        });

        tbody.find(".table-row-item").on("click", function () {
            const rowId = $(this).data("id");
            const rowLat = parseFloat($(this).data("lat"));
            const rowLng = parseFloat($(this).data("lng"));

            map.setView([rowLat, rowLng], 15);

            const targetLayer = layers.find((l) => {
                const props = l.feature
                    ? l.feature.properties
                    : l.options.properties || {};
                return props.id == rowId;
            });

            if (targetLayer) {
                featureGroups[type].fire("click", {
                    latlng: targetLayer.getLatLng(),
                    layer: targetLayer,
                });
            }
        });
    }

    // تعریف آیکون‌های سفارشی برای هر نوع نقطه
    const nodeIcons = {
        dc: dc_icon,
        da: da_icon,
        ec: ec_icon,
        tmc: tmc_icon,
        h: h_icon,
    };

    const pulsingIcon = puls_icon;

    // Loading HSC nodes and parameters when entering the app
    const featureGroups = {
        dc: new L.FeatureGroup(),
        ec: new L.FeatureGroup(),
        da: new L.FeatureGroup(),
        tmc: new L.FeatureGroup(),
        h: new L.FeatureGroup(),
    };

    let geojsonNodes = null;
    let hscParameters = null;
    var pulish_marker = new L.Marker();

    axios
        .get("http://127.0.0.1:8000/load_data")
        .then((response) => {
            geojsonNodes = response.data.geojson_nodes;
            hscParameters = response.data.hsc_parameters;
            L.geoJSON(geojsonNodes, {
                pointToLayer: (feature, latlng) => {
                    const type = feature.properties.type;
                    const icon = nodeIcons[type] ?? defaultIcon;
                    const marker = L.marker(latlng, {
                        icon: icon,
                        properties: feature.properties,
                    });
                    if (featureGroups[type]) {
                        marker.addTo(featureGroups[type]).addTo(map);
                    }
                    return marker;
                },
            });
        })
        .then(() => {
            Object.entries(featureGroups).forEach(([type, featureGroup]) => {
                featureGroup.on("click", (e) => {
                    if (
                        $("#edit-btn").hasClass("disabled") &&
                        $("#delete-btn").hasClass("disabled")
                    ) {
                        $("#edit-btn").removeClass("disabled");
                        $("#delete-btn").removeClass("disabled");
                    }
                    if (!$("#save-actions").hasClass("d-none")) {
                        $("#save-actions").addClass("d-none");
                        $("#edit-actions").removeClass("d-none");
                        cancel_btn_event();
                    }
                    pulish_marker.removeFrom(map);
                    pulish_marker = L.marker(e.latlng, {
                        icon: pulsingIcon,
                        pane: "backgroundMarkers",
                    }).addTo(map);
                    let lat = $("#generic_prop").find(".lat");
                    let lng = $("#generic_prop").find(".lng");
                    lat.val(e.latlng.lat);
                    lat[0].defaultValue = e.latlng.lat;
                    lng.val(e.latlng.lng);
                    lng[0].defaultValue = e.latlng.lng;
                    const props = e.layer.feature.properties;
                    const node_type = e.layer.feature.properties.type;
                    $(".props").children().addClass("d-none");
                    $(`.${node_type}`).removeClass("d-none");
                    node_prop_display (node_type, props);
                });
            });
        });

    function node_prop_display(node_type, props) {
        Object.entries(props).forEach(([prop, val]) => {
            var elementTag = $("#generic_prop")
                .find(`.${node_type}`)
                .find(`.${prop}`);
            if (elementTag.length === 0) {
                elementTag = $("#generic_prop").find(`.${prop}`);
            }
            if (elementTag.is("textarea") || elementTag.is("input")) {
                elementTag.val(val);
                if (elementTag[0]) elementTag[0].defaultValue = val;
            } else {
                elementTag.text(val);
                if (elementTag[0]) elementTag[0].defaultValue = val;
            }
        });
    }

    $("#edit-btn").on("click", function (e) {
        e.preventDefault(); // Prevent default form submission
        edit_btn_event();
    });
    $("#delete-btn").on("click", function (e) {
        e.preventDefault(); // Prevent default form submission
        delete_btn_event();
    });

    $("#cancel-btn").on("click", function (e) {
        e.preventDefault(); // Prevent default form submission
        cancel_btn_event();
    });
    $("#save-btn").on("click", function (e) {
        e.preventDefault(); // Prevent default form submission
        save_btn_event();
    });

    function edit_btn_event() {
        let El_actions = $("#edit-btn").parents(".actions");
        El_actions.children().addClass("d-none");
        El_actions.find("#save-actions").removeClass("d-none");
        $(".node-props")
            .find("input, textarea")
            .each(function () {
                $(this).prop("disabled", false);
            });
    }

    function cancel_btn_event() {
        let El_actions = $("#cancel-btn").parents(".actions");
        El_actions.children().addClass("d-none");
        El_actions.find("#edit-actions").removeClass("d-none");
        $(".node-props")
            .find("input, textarea")
            .each(function () {
                // ۱. بازگرداندن مقدار به مقدار پیش‌فرض اولیه
                this.value = this.defaultValue;
                // ۲. غیرفعال کردن فیلد
                $(this).prop("disabled", true);
            });
    }

    function save_btn_event() {
        const form = $(".node-card");
        const idRaw = form.find(".id").text();
        const id = idRaw && idRaw !== "-" ? idRaw : null;
        const type = form.find(".type").text();
        const url = id ? `/nodes/${id}` : "/nodes";
        const method = id ? "PUT" : "POST";

        const data = {};
        form.find("*").not(":hidden").serializeArray().forEach((item) => {
            data[item.name] = item.value;
        });

        console.log(data);
        data.id = id;
        data.type = type;

        axios({
            method: method,
            url: url,
            data: data,
        })
            .then((response) => {
                if (response.data.success) {
                    alert("Saved successfully!");
                    // Update map and table without full page reload
                    updateNodeOnMap(data.id, data.type, data);
                    populateTable(data.type);
                    cancel_btn_event(); // Revert to disabled state after save
                } else {
                    alert("Error: " + response.data.message);
                }
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while saving.");
            });
    }

    function delete_btn_event() {
        const idRaw = $("#generic_prop").find(".id").text();
        const id = idRaw && idRaw !== "-" ? idRaw : null;
        const type = $("#generic_prop").find(".type").text();
        if (!id) {
            alert("No node selected for deletion.");
            return;
        }

        if (!confirm("Are you sure you want to delete this node?")) {
            return;
        }

        axios
            .delete(`/nodes/${id}`)
            .then((response) => {
                if (response.data.success) {
                    alert("Deleted successfully!");
                    removeNodeFromMap(id, type);
                    populateTable(type);
                    resetNodeProperties(); // Clear form and disable inputs/buttons
                } else {
                    alert("Error: " + response.data.message);
                }
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while deleting.");
            });
    }

    function resetNodeProperties() {
        $("#edit-btn").addClass("disabled");
        $("#delete-btn").addClass("disabled");

        $("#generic_prop").find(".id").text("-");
        $("#generic_prop").find(".type").text("-");

        $(".node-props")
            .find("input, textarea")
            .each(function () {
                this.value = "";
                this.defaultValue = "";
                $(this).prop("disabled", true);
            });

        $(".props").children().addClass("d-none");
    }

    function updateNodeOnMap(id, type, newData) {
        featureGroups[type].eachLayer(function (layer) {
            if (layer.feature && layer.feature.properties.id == id) {
                // Update properties
                for (const key in newData) {
                    if (newData.hasOwnProperty(key)) {
                        layer.feature.properties[key] = newData[key];
                    }
                }
                // Update latlng if changed
                if (newData.lat && newData.lng) {
                    layer.setLatLng([newData.lat, newData.lng]);
                    pulish_marker.removeFrom(map);
                    pulish_marker = L.marker(layer.getLatLng(), {
                        icon: pulsingIcon,
                        pane: "backgroundMarkers",
                    }).addTo(map);
                }
                // Optionally, update popup content here if needed
                let props = layer.feature.properties;
                console.log(props);
                node_prop_display(type, props);
            }
        });
    }

    function removeNodeFromMap(id, type) {
        featureGroups[type].eachLayer(function (layer) {
            if (layer.feature && layer.feature.properties.id == id) {
                featureGroups[type].removeLayer(layer);
                map.removeLayer(layer);
            }
        });
        pulish_marker.removeFrom(map);
    }
});
