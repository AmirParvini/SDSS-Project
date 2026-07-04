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
                props.lat = latlng.lat.toFixed(2);
            }
            if (props.lng === undefined) {
                props.lng = latlng.lng.toFixed(2);
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
            var pulish_marker = new L.Marker();
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
                    $("#generic_prop").find(".lat").val(e.latlng.lat);
                    $("#generic_prop").find(".lng").val(e.latlng.lng);
                    const props = e.layer.feature.properties;
                    const node_type = e.layer.feature.properties.type;
                    $(".props").children().addClass("d-none");
                    $(`.${node_type}`).removeClass("d-none");
                    Object.entries(props).forEach(([prop, val]) => {
                        var elementTag = $("#generic_prop").find(`.${prop}`);
                        if (
                            elementTag.is("textarea") ||
                            elementTag.is("input")
                        ) {
                            elementTag.val(val);
                        } else {
                            elementTag.text(val);
                        }
                    });
                });
            });
        });

    $("#edit-btn").on("click", function () {
        edit_btn_event();
    });
    $("#cancel-btn").on("click", function () {
        delete_btn_event();
    });
    
    $("#cancel-btn").on("click", function () {
        cancel_btn_event();
    });
    $("#cancel-btn").on("click", function () {
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
                $(this).prop("disabled", true);
            });
    }

    // let polyline = L.polyline(path, {
    //         color: 'red',       // رنگ خط
    //         weight: 5,          // ضخامت خط
    //         opacity: 0.7        // شفافیت
    //     }).addTo(map);
    //     map.fitBounds(polyline.getBounds());
    // let SP = new ShowPoints();
    // const points_json = [];
    // SP.loadpoints(Points); //Points sent from home.blade
    // Object.entries(Points).forEach(([pointType, points]) => {
    //     if (points && Array.isArray(points)) {
    //         points.forEach(point => {
    //             points_json.push({
    //                 type: pointType,
    //                 point: point
    //             });
    //         });
    //     }
    // });
    // console.log('points_json:', points_json);
    // // Map click event to add point
    // const modal = new bootstrap.Modal($("#addPointModal"));
    // map.on("click", function (e) {
    //     $("#pointLat").val(e.latlng.lat);
    //     $("#pointLng").val(e.latlng.lng);
    //     modal.show();
    // });

    // const elementVisibility = SP.elementVisibility;
    // $(".form-select").on("change", function () {
    //     let selectedValue = $(this).val();
    //     let visibility = elementVisibility[selectedValue];
    //     if (visibility) {
    //         Object.keys(visibility).forEach(key => {
    //             const element = $("#" + key);
    //             if (visibility[key]) {
    //                 element.addClass("d-none");
    //             } else {
    //                 element.removeClass("d-none");
    //             }
    //         });
    //     }
    // });

    // const points_icon = SP.iconMap;

    // // Save point
    // const params = {
    //         IDC: { capacity: "#idc_Capacity", cost: "#pointCost"},
    //         EC: { demand: "#pointDemand"},
    //         TMC: { capacity: "#h_tmc_Capacity", cost: "#pointCost"},
    //         H: { capacity: "#h_tmc_Capacity"},
    //         DA: { injured: "#pointInjured"}
    //     };
    // $("#savePoint").on("click", function () {
    //     const formData = new FormData();
    //     formData.append("name", $("#pointName").val());
    //     formData.append("lat", $("#pointLat").val());
    //     formData.append("lng", $("#pointLng").val());
    //     formData.append("_token", $('meta[name="csrf-token"]').attr("content"));
    //     let fields = params[$(".form-select").val()];
    //     Object.keys(fields).forEach(key => {
    //         console.log(key, $(fields[key]).val());
    //         formData.append(key, $(fields[key]).val());
    //     });
    //     fetch(`/${$(".form-select").val()}`, {
    //         method: "POST",
    //         body: formData,
    //     })
    //         .then((response) => {
    //             if (!response.ok) {
    //                 return response.text().then((text) => {
    //                     throw new Error(`HTTP ${response.status}: ${text}`);
    //                 });
    //             }
    //             return response.json();
    //         })
    //         .then((data) => {
    //             if (data.success) {
    //                 // Add marker to map
    //                 const lat = $("#pointLat").val();
    //                 const lng = $("#pointLng").val();
    //                 const name = $("#pointName").val();
    //                 console.log(points_icon[$(".form-select").val()])
    //                 const marker = L.marker([lat, lng], {icon: points_icon[$(".form-select").val()]})
    //                     .addTo(markerGroup);
    //                 // Add click event to the marker
    //                 const point = data.point;
    //                 const pointType = data.pointtype
    //                 marker.id = point.id;
    //                 SP.markersById[point.id] = marker;
    //                 console.log('point.id:', point.id);
    //                 marker.on('click', function(e) {
    //                     SP.editmodal.show();
    //                     SP.selectmarkerid = point.id;
    //                     SP.selectmarkertype = pointType;
    //                     $("#pointType .form-control").val(
    //                         SP.typeNames[pointType]
    //                     );
    //                     $("#pointNameEdit .form-control").val(point.name);
    //                     $("#pointLatEdit .form-control").val(point.lat);
    //                     $("#pointLngEdit .form-control").val(point.lng);
    //                     $(".showinput").hide();
    //                     Object.entries(SP.params[pointType]).forEach(
    //                         ([param, elementid]) => {
    //                             $("#pointType").show();
    //                             $("#pointNameEdit").show();
    //                             $("#pointLatEdit").show();
    //                             $("#pointLngEdit").show();
    //                             $(elementid).show();
    //                             $(`${elementid} .form-control`).val(
    //                                 point[param]
    //                             );
    //                         }
    //                     );
    //                 });
    //                 modal.hide();
    //                 $("#addPointForm")[0].reset();
    //                 let popupContent = `<div dir="rtl" class="text-center">`;
    //                     popupContent += `<b>${pointType}</b><br>`;
    //                     popupContent += `<b>نام:</b> ${
    //                         point.name || "نامشخص"
    //                     }<br>`;
    //                     // Add specific information based on point type
    //                     if (pointType === "da_points" && point.injured) {
    //                         popupContent += `<b>تعداد مجروحین:</b> ${point.injured}<br>`;
    //                     } else if (pointType === "ec_points" && point.demand) {
    //                         popupContent += `<b>تقاضا:</b> ${point.demand}<br>`;
    //                     } else if (
    //                         (pointType === "idc_points" ||
    //                             pointType === "tmc_points" ||
    //                             pointType === "H_points") &&
    //                         point.capacity
    //                     ) {
    //                         popupContent += `<b>ظرفیت:</b> ${point.capacity}<br>`;
    //                     }

    //                     if (point.cost) {
    //                         popupContent += `<b>هزینه:</b> ${point.cost} دلار<br>`;
    //                     }

    //                     popupContent += `</div>`;

    //                     marker.bindPopup(popupContent);

    //                     // Add hover effects
    //                     marker.on("mouseover", function () {
    //                         this.openPopup();
    //                     });

    //                     marker.on("mouseout", function () {
    //                         this.closePopup();
    //                     });
    //             } else {
    //                 alert(
    //                     "Error saving point: " +
    //                         (data.message || JSON.stringify(data.errors))
    //                 );
    //             }
    //         })
    //         .catch((error) => {
    //             console.error("Error:", error);
    //             alert("Error saving point");
    //         });
    // });

    // $("#deletePoint").on("click", function () {
    //     console.log("Deleting point:", SP.selectmarkertype, SP.selectmarkerid);
    //     fetch(`/${SP.selectmarkertype}/${SP.selectmarkerid}`, {
    //         method: "DELETE",
    //         headers: {
    //             "X-CSRF-TOKEN": document
    //                 .querySelector('meta[name="csrf-token"]')
    //                 .getAttribute("content"),
    //             "Content-Type": "application/json",
    //         },
    //     })
    //         .then((response) => {
    //             if (!response.ok) {
    //                 return response.text().then((text) => {
    //                     throw new Error(`HTTP ${response.status}: ${data.message}`);
    //                 });
    //             }
    //             return response.json();
    //         })
    //         .then((data) => {
    //             if (data.success) {
    //                 try {
    //                     markerGroup.removeLayer(
    //                         SP.markersById[SP.selectmarkerid]
    //                     );
    //                     delete SP.markersById[SP.selectmarkerid];
    //                     SP.editmodal.hide();
    //                     alert("Point deleted successfully");
    //                 } catch (e) {
    //                     console.error("Error removing marker:", e);
    //                 }
    //             }
    //         });
    // });
});
