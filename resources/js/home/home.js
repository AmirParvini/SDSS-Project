import { map, minimap, map_tiles, markerGroup } from "./map_init.js";
import ShowPoints from "./show_points.js";
$(function () {
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

    // fade-in/out itmes table
    $(".imgitems").on("click", function (e) {
        if ($(".itemstable").hasClass("d-none")) {
            $(".itemstable").removeClass("fade-out-left d-none");
            setTimeout(() => {
                $(".itemstable").addClass("moved");
            }, 10);
        } else {
            $(".itemstable").addClass("fade-out-left");
            setTimeout(() => {
                $(".itemstable").addClass("d-none");
            }, 100);
            $(".itemstable").removeClass("moved");
        }
    });
});
