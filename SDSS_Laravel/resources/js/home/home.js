import {
    map,
    markerGroup,
} from "./map_init.js";
import ShowPoints from "./show_points.js";

$(document).ready(function () {
//     var path = [
//         [
//   [35.740951, 51.4168272],
//   [35.7412484, 51.4164813],
//   [35.7414909, 51.4162002],
//   [35.7415683, 51.4161212],
//   [35.741676, 51.4160291],
//   [35.7417968, 51.4159454],
//   [35.7419233, 51.415875],
//   [35.7419516, 51.4158205],
//   [35.7419723, 51.4157752],
//   [35.7419823, 51.4157246],
//   [35.7419766, 51.4156662],
//   [35.74195, 51.4156147],
//   [35.7419171, 51.4155818],
//   [35.7418635, 51.4155392],
//   [35.7412397, 51.4155875],
//   [35.7411731, 51.4155973],
//   [35.739594, 51.415832],
//   [35.7393673, 51.4158687],
//   [35.7391369, 51.4158992],
//   [35.7386811, 51.4159419],
//   [35.7385695, 51.4159506],
//   [35.7384628, 51.4159594],
//   [35.7383667, 51.4159596],
//   [35.7383033, 51.4159558],
//   [35.7382324, 51.4159448],
//   [35.7381615, 51.4159292],
//   [35.7380821, 51.4159132],
//   [35.7380404, 51.415901],
//   [35.7380053, 51.4158883],
//   [35.7379774, 51.415876],
//   [35.7379572, 51.4158647],
//   [35.7378363, 51.4157833],
//   [35.7377335, 51.415709],
//   [35.7377014, 51.4156804],
//   [35.737553, 51.4155201],
//   [35.7375262, 51.4154993],
//   [35.7374971, 51.4154803],
//   [35.7373886, 51.415464],
//   [35.7373528, 51.4153368],
//   [35.7373001, 51.4152487],
//   [35.7372446, 51.4151783],
//   [35.7371876, 51.4151259],
//   [35.7371197, 51.4150777],
//   [35.7370691, 51.4150608],
//   [35.7370236, 51.4150531],
//   [35.7369523, 51.4150696],
//   [35.7368979, 51.4150961],
//   [35.7364733, 51.415247],
//   [35.7361364, 51.4153641],
//   [35.735648, 51.4155677],
//   [35.7350469, 51.4133153],
//   [35.735021, 51.4132139],
//   [35.7348551, 51.4132708],
//   [35.7342875, 51.4134653],
//   [35.7342032, 51.4134942],
//   [35.7339348, 51.4135861],
//   [35.733353, 51.4137855],
//   [35.733029, 51.4138965],
//   [35.732387, 51.4141165],
//   [35.732085, 51.41422],
//   [35.7316424, 51.4143716],
//   [35.7315221, 51.4144128],
//   [35.7310038, 51.4145904],
//   [35.7308984, 51.4146265],
//   [35.7304943, 51.414765],
//   [35.730196, 51.4148672],
//   [35.7301112, 51.4148962],
//   [35.729048, 51.4152605],
//   [35.7286497, 51.415397],
//   [35.7278446, 51.4156728],
//   [35.7277064, 51.4157202],
//   [35.7276688, 51.415585],
//   [35.7275564, 51.4151801],
//   [35.7272149, 51.4138821],
//   [35.727146, 51.4137329],
//   [35.7270973, 51.4136435],
//   [35.7270712, 51.4136113],
//   [35.7270567, 51.4135956],
//   [35.72704, 51.4135775],
//   [35.7269334, 51.4134981],
//   [35.7264478, 51.4131916],
//   [35.7262448, 51.4130348],
//   [35.7261242, 51.4129422],
//   [35.7257675, 51.4126021],
//   [35.7250353, 51.411753],
//   [35.7242149, 51.4108122],
//   [35.7241396, 51.4107218],
//   [35.7239233, 51.4104628],
//   [35.7238938, 51.4104152],
//   [35.7238102, 51.4102801],
//   [35.7237704, 51.4102158],
//   [35.7236215, 51.4100939],
//   [35.7235865, 51.4100556],
//   [35.7234316, 51.4098974],
//   [35.7231569, 51.4096917],
//   [35.7227568, 51.4094177],
//   [35.722765, 51.4093029],
//   [35.7227645, 51.4092797],
//   [35.7227614, 51.4092578],
//   [35.7226542, 51.4085443],
//   [35.7232529, 51.4084122],
//   [35.7233307, 51.4084251],
//   [35.7239916, 51.4082793],
//   [35.7239834, 51.4082071],
//   [35.7238814, 51.4073087],
//   [35.723748, 51.4061778],
//   [35.7237417, 51.4061138],
//   [35.7237352, 51.4060233],
//   [35.7237079, 51.4050585],
//   [35.7236819, 51.4040852],
//   [35.7236771, 51.4039522],
//   [35.7236777, 51.4025117],
//   [35.7240223, 51.4025032],
//   [35.7240428, 51.4025007],
//   [35.7240738, 51.4024892],
//   [35.7240963, 51.4024759],
//   [35.7241138, 51.402464],
//   [35.7241301, 51.4024449],
//   [35.724142, 51.4024186],
//   [35.7241533, 51.4023826],
//   [35.7241571, 51.402358],
//   [35.7241608, 51.4023264],
//   [35.7241591, 51.402279],
//   [35.7241547, 51.4022055],
//   [35.724151, 51.4021816],
//   [35.7241492, 51.4021697],
//   [35.7241389, 51.4021429],
//   [35.7241271, 51.4021215],
//   [35.7241118, 51.4021054],
//   [35.7240978, 51.4020928],
//   [35.7240825, 51.4020827],
//   [35.7240672, 51.4020762],
//   [35.7240431, 51.4020722],
//   [35.7236473, 51.4020852],
//   [35.723615, 51.4007152],
//   [35.7236094, 51.4002625],
//   [35.7236281, 51.4000373],
//   [35.723642, 51.3999466],
//   [35.7236744, 51.3998676],
//   [35.7237065, 51.3997947],
//   [35.7238507, 51.3995944],
//   [35.723976, 51.3994324],
//   [35.7241186, 51.3992013],
//   [35.7242782, 51.398856],
//   [35.7243685, 51.3987297],
//   [35.7245826, 51.3983522],
//   [35.7246734, 51.398131],
//   [35.724737, 51.39797],
//   [35.7248178, 51.3976632],
//   [35.7248599, 51.3974556],
//   [35.7248727, 51.3971008],
//   [35.7248647, 51.3969658],
//   [35.7248553, 51.3967524],
//   [35.7248403, 51.3965568],
//   [35.7248183, 51.3963862],
//   [35.7247876, 51.3962297],
//   [35.7247511, 51.3960713],
//   [35.7247059, 51.3959236],
//   [35.7246077, 51.3956716],
//   [35.7244372, 51.3952912],
//   [35.7240136, 51.3944915],
//   [35.7236827, 51.3938204],
//   [35.7235631, 51.3934873],
//   [35.7235048, 51.3933267],
//   [35.7234295, 51.3930999],
//   [35.7233039, 51.3925605],
//   [35.7232531, 51.3922343],
//   [35.723225, 51.3919512],
//   [35.7232201, 51.3918708],
//   [35.7232166, 51.3918048],
//   [35.7231946, 51.3907373],
//   [35.7231717, 51.3883797],
//   [35.7231974, 51.3879113],
//   [35.7232396, 51.3874804],
//   [35.7232837, 51.3871146],
//   [35.723371, 51.3865333],
//   [35.7234647, 51.386066],
//   [35.7235543, 51.3857027],
//   [35.7237312, 51.3851584],
//   [35.7238663, 51.3846028],
//   [35.7239194, 51.3844237],
//   [35.7239777, 51.384227],
//   [35.7241252, 51.383859],
//   [35.7243233, 51.3833274],
//   [35.7244078, 51.383094],
//   [35.7245547, 51.3827075],
//   [35.7247737, 51.3823015],
//   [35.7247697, 51.3822291],
//   [35.7247652, 51.3821993],
//   [35.7247548, 51.38216],
//   [35.7247331, 51.3821122],
//   [35.7247081, 51.3820885],
//   [35.7246772, 51.3820748],
//   [35.7246397, 51.3820686],
//   [35.724609, 51.3820713],
//   [35.7245812, 51.3820784],
//   [35.7245441, 51.3820995],
//   [35.7245281, 51.3821134],
//   [35.7245145, 51.3821309],
//   [35.7243652, 51.3823904],
//   [35.7242004, 51.3827648],
//   [35.7240352, 51.3831858],
//   [35.7239922, 51.3832853],
//   [35.7238562, 51.3835696],
//   [35.7237195, 51.383852],
//   [35.7235365, 51.3842403],
//   [35.7234879, 51.3843438],
//   [35.7233452, 51.3846199],
//   [35.7232427, 51.3848202],
//   [35.7231822, 51.3849612],
//   [35.7230599, 51.3852766],
//   [35.7230325, 51.3853303],
//   [35.7225564, 51.3862504],
//   [35.7223621, 51.386626],
//   [35.7221988, 51.3869539],
//   [35.7221075, 51.3871373],
//   [35.7214835, 51.3883429]
// ]
// ]
// let polyline = L.polyline(path, {
//         color: 'red',       // رنگ خط
//         weight: 5,          // ضخامت خط
//         opacity: 0.7        // شفافیت
//     }).addTo(map);
//     map.fitBounds(polyline.getBounds());
    let SP = new ShowPoints();
    const points_json = [];
    SP.loadpoints(Points); //Points sent from home.blade
    Object.entries(Points).forEach(([pointType, points]) => {
        if (points && Array.isArray(points)) {
            points.forEach(point => {
                points_json.push({
                    type: pointType,
                    point: point
                });
            });
        }
    });
    console.log('points_json:', points_json);
    // Map click event to add point
    const modal = new bootstrap.Modal($("#addPointModal"));
    map.on("click", function (e) {
        $("#pointLat").val(e.latlng.lat);
        $("#pointLng").val(e.latlng.lng);
        modal.show();
    });

    const elementVisibility = SP.elementVisibility;
    $(".form-select").on("change", function () {
        let selectedValue = $(this).val();
        let visibility = elementVisibility[selectedValue];
        if (visibility) {
            Object.keys(visibility).forEach(key => {
                const element = $("#" + key);
                if (visibility[key]) {
                    element.addClass("d-none");
                } else {
                    element.removeClass("d-none");
                }
            });
        }
    });

    const points_icon = SP.iconMap;

    // Save point
    const params = {
            IDC: { capacity: "#idc_Capacity", cost: "#pointCost"},
            EC: { demand: "#pointDemand"},
            TMC: { capacity: "#h_tmc_Capacity", cost: "#pointCost"},
            H: { capacity: "#h_tmc_Capacity"},
            DA: { injured: "#pointInjured"}
        };
    $("#savePoint").on("click", function () {
        const formData = new FormData();
        formData.append("name", $("#pointName").val());
        formData.append("lat", $("#pointLat").val());
        formData.append("lng", $("#pointLng").val());
        formData.append("_token", $('meta[name="csrf-token"]').attr("content"));
        let fields = params[$(".form-select").val()];
        Object.keys(fields).forEach(key => {
            console.log(key, $(fields[key]).val());
            formData.append(key, $(fields[key]).val());
        });
        fetch(`/${$(".form-select").val()}`, {
            method: "POST",
            body: formData,
        })
            .then((response) => {
                if (!response.ok) {
                    return response.text().then((text) => {
                        throw new Error(`HTTP ${response.status}: ${text}`);
                    });
                }
                return response.json();
            })
            .then((data) => {
                if (data.success) {
                    // Add marker to map
                    const lat = $("#pointLat").val();
                    const lng = $("#pointLng").val();
                    const name = $("#pointName").val();
                    console.log(points_icon[$(".form-select").val()])
                    const marker = L.marker([lat, lng], {icon: points_icon[$(".form-select").val()]})
                        .addTo(markerGroup);
                    // Add click event to the marker
                    const point = data.point;
                    const pointType = data.pointtype
                    marker.id = point.id;
                    SP.markersById[point.id] = marker;
                    console.log('point.id:', point.id);
                    marker.on('click', function(e) {
                        SP.editmodal.show();
                        SP.selectmarkerid = point.id;
                        SP.selectmarkertype = pointType;
                        $("#pointType .form-control").val(
                            SP.typeNames[pointType]
                        );
                        $("#pointNameEdit .form-control").val(point.name);
                        $("#pointLatEdit .form-control").val(point.lat);
                        $("#pointLngEdit .form-control").val(point.lng);
                        $(".showinput").hide();
                        Object.entries(SP.params[pointType]).forEach(
                            ([param, elementid]) => {
                                $("#pointType").show();
                                $("#pointNameEdit").show();
                                $("#pointLatEdit").show();
                                $("#pointLngEdit").show();
                                $(elementid).show();
                                $(`${elementid} .form-control`).val(
                                    point[param]
                                );
                            }
                        );
                    });
                    modal.hide();
                    $("#addPointForm")[0].reset();
                    let popupContent = `<div dir="rtl" class="text-center">`;
                        popupContent += `<b>${pointType}</b><br>`;
                        popupContent += `<b>نام:</b> ${
                            point.name || "نامشخص"
                        }<br>`;
                        // Add specific information based on point type
                        if (pointType === "da_points" && point.injured) {
                            popupContent += `<b>تعداد مجروحین:</b> ${point.injured}<br>`;
                        } else if (pointType === "ec_points" && point.demand) {
                            popupContent += `<b>تقاضا:</b> ${point.demand}<br>`;
                        } else if (
                            (pointType === "idc_points" ||
                                pointType === "tmc_points" ||
                                pointType === "H_points") &&
                            point.capacity
                        ) {
                            popupContent += `<b>ظرفیت:</b> ${point.capacity}<br>`;
                        }

                        if (point.cost) {
                            popupContent += `<b>هزینه:</b> ${point.cost} دلار<br>`;
                        }

                        popupContent += `</div>`;

                        marker.bindPopup(popupContent);

                        // Add hover effects
                        marker.on("mouseover", function () {
                            this.openPopup();
                        });

                        marker.on("mouseout", function () {
                            this.closePopup();
                        });
                } else {
                    alert(
                        "Error saving point: " +
                            (data.message || JSON.stringify(data.errors))
                    );
                }
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("Error saving point");
            });
    });

    $("#deletePoint").on("click", function () {
        console.log("Deleting point:", SP.selectmarkertype, SP.selectmarkerid);
        fetch(`/${SP.selectmarkertype}/${SP.selectmarkerid}`, {
            method: "DELETE",
            headers: {
                "X-CSRF-TOKEN": document
                    .querySelector('meta[name="csrf-token"]')
                    .getAttribute("content"),
                "Content-Type": "application/json",
            },
        })
            .then((response) => {
                if (!response.ok) {
                    return response.text().then((text) => {
                        throw new Error(`HTTP ${response.status}: ${data.message}`);
                    });
                }
                return response.json();
            })
            .then((data) => {
                if (data.success) {
                    try {
                        markerGroup.removeLayer(
                            SP.markersById[SP.selectmarkerid]
                        );
                        delete SP.markersById[SP.selectmarkerid];
                        SP.editmodal.hide();
                        alert("Point deleted successfully");
                    } catch (e) {
                        console.error("Error removing marker:", e);
                    }
                }
            });
    });

    let controller;
    let isFetching = false;

    // $('#run').on('click', function(){
    //     $('#commo_shipped_table').addClass('d-none');
    //     let district = $('#district').text();
    //     let APR = $('#APR').val();
    //     let PP = $('#PP').text();
    //     let Config = $('#Config').text();
    //     if(isFetching == false){
    //         isFetching = true;
    //         $(this).html('stop');
    //         $('#error_message').html('');
    //         controller = new AbortController();
    //         const signal = controller.signal;
    //         fetch('/solving', {
    //             signal,
    //             method: 'POST',
    //             headers: {
    //                 'Content-Type': 'application/json',
    //                 'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content'),
    //             },
    //             body: JSON.stringify({
    //                 District: district,
    //                 APR: APR,
    //                 PP: PP,
    //                 Config: Config,
    //             }),
    //         })
    //         .then(response => {
    //             if(!response.ok){
    //                 console.log('error:', response);
    //             }
    //             return response.json();
    //         })
    //         .then(data => {
    //             if(data.status == 'success'){
    //                 console.log(data);
    //                 const keys = Object.keys(data.output);
    //                 const values = Object.values(data.output);
    //                 console.log('keys: ', keys);
    //                 console.log('values: ', values);
    //                 FetchedDataParse(values);
    //             } else if(data.status == 'error'){
    //                 console.log('error:', data);
    //                 let errorMsg = data.message || 'An error occurred';
    //                 $('#error_message').html(errorMsg);
    //             }
    //             $(this).html('Solve Model');
    //             isFetching = false;
    //         })
    //         .catch(error => {
    //             if(error.name === 'AbortError'){
    //                 $('#error_message').html('Fetch request was canceled.');
    //             } else {
    //                 $('#error_message').html('Fetch failed.');
    //                 console.error('Fetch error:', error);
    //             }
    //             $(this).html('Solve Model');
    //             isFetching = false;
    //         });
    //     } else {
    //         isFetching = false;
    //         controller.abort();
    //         $(this).html('Solve Model');
    //     }
    // });

    // var ec_demands = {};
    // var ldc_inventories = {};
    // var inventory_report_values = {};
    // var service_report_values = {};
    // function FetchedDataParse(values){
    //     let solution_report_id_array = [
    //         'Solution Status',
    //         'Iterations',
    //         'Solution Time (sec)',
    //         'Total Distance',
    //         'Total LDC Used',
    //         'Total Unmet Demand Amount',
    //         'Total Additional Inventory',
    //     ];
    //     const solution_report_values = values[0];
    //     inventory_report_values = values[1];
    //     service_report_values = values[2];
    //     const nodes_coordinate_report_values = values[3];
    //     const additional_unmetdemand_report_values = values[4];
    //     const ec_name_unmetdemand = Object(additional_unmetdemand_report_values.EC_name_unmetdemand);
    //     const ldc_name_additional_inventory = Object(additional_unmetdemand_report_values.LDC_name_additional_inventory);
    //     show_nodes(nodes_coordinate_report_values);
    //     // filling Solution Information table
    //     Object.entries(solution_report_values).forEach(([key, value], index) => {
    //         const $report_row = $('#' + solution_report_id_array[index]);
    //         $report_row.html('');
    //         $report_row.html(value);
    //     });
    //     // filling inventory decision table
    //     filling_table(inventory_report_values, 'inventory_table');
    //     // filling service decision table
    //     filling_table(service_report_values, 'service_table');
    //     // filling additional inventory table
    //     filling_additional_unmetdemand_table(ldc_name_additional_inventory, 'additional_inventory_table');
    //     // filling unmet demand table
    //     filling_additional_unmetdemand_table(ec_name_unmetdemand, 'unmet_demand_table');

    // }

    // var CMD_name_coordinates = {};
    // var LDC_name_coordinates = {};
    // var EC_name_coordinates = {};
    // var Nodes_coordinate = {};

    // function show_nodes(nodes_coordinate_report_values){
    //     markerGroup.clearLayers();
    //     lineGroup.clearLayers();
    //     const CMD_coordinate = [];
    //     const LDC_coordinate = [];
    //     const EC_coordinate = [];
    //     CMD_name_coordinates = nodes_coordinate_report_values.CMDs_coordinate;
    //     LDC_name_coordinates = nodes_coordinate_report_values.LDCs_coordinate;
    //     EC_name_coordinates = nodes_coordinate_report_values.ECs_coordinate;
    //     Nodes_coordinate = {...CMD_name_coordinates, ...LDC_name_coordinates, ...EC_name_coordinates};
    //     Object.entries(CMD_name_coordinates).forEach(([key, value]) => {
    //         const cmd_allocated = inventory_report_values[key];
    //         cmd_allocated.forEach((ldc_info) => {
    //             const ldc_name = ldc_info[0];
    //             const ldc_inventory = ldc_info[1];
    //             ldc_inventories[ldc_name] = ldc_inventory;
    //         });
    //         const cmd_marker = L.marker([value[0], value[1]])
    //             .addTo(markerGroup)
    //             .bindPopup('<b>نام پایگاه امداد: </b>' + key);
    //         cmd_marker.on('mouseover', function(){
    //             this.openPopup();
    //         });
    //         cmd_marker.on('mouseout', function(){
    //             this.closePopup();
    //         });
    //         cmd_marker.on('click', function(){
    //             lineGroup.clearLayers();
    //             commodity_shipped_table('commo_shipped_table', 'inventory_table', key);
    //             cmd_allocated.forEach((ldc_info) => {
    //                 const ldc_name = ldc_info[0];
    //                 const line = [
    //                     CMD_name_coordinates[key],
    //                     LDC_name_coordinates[ldc_name],
    //                 ];
    //                 L.polyline(line, {color: 'black'}).addTo(lineGroup);
    //             });
    //         });
    //     });
    //     Object.entries(LDC_name_coordinates).forEach(([key, value]) => {
    //         const ldc_allocated = service_report_values[key];
    //         ldc_allocated.forEach((ec_info) => {
    //             const ec_name = ec_info[0];
    //             const ec_demand = ec_info[1];
    //             if(!ec_demands[ec_name]){
    //                 ec_demands[ec_name] = [];
    //             }
    //             ec_demands[ec_name].push(ec_demand);
    //         });
    //         const ldc_marker = L.marker([value[0], value[1]])
    //             .addTo(markerGroup)
    //             .bindPopup(
    //                 '<div class="lrt"><br><b>نام مرکز توزیع محلی: </b>' +
    //                     key + '<br>' +
    //                     '<br>مقدار موجودی آب: ' +
    //                     ldc_inventories[key][0].toFixed(2) +
    //                     '<br>مقدار موجودی غذا: ' +
    //                     ldc_inventories[key][1].toFixed(2) +
    //                     '<br>مقدار موجودی کیت پزشکی: ' +
    //                     ldc_inventories[key][2].toFixed(2) +
    //                     '<br>مقدار موجودی چادر امداد: ' +
    //                     ldc_inventories[key][3].toFixed(2) + '<br>' +
    //                     '<br>مجموع موجودی کالا‌ها: ' +
    //                     ldc_inventories[key].reduce((a, b) => a + b, 0).toFixed(2) + '</div>'
    //             )
    //             .setIcon(blueCircle);
    //         ldc_marker.on('mouseover', function(){
    //             this.openPopup();
    //         });
    //         ldc_marker.on('mouseout', function(){
    //             this.closePopup();
    //         });
    //         ldc_marker.on('click', function(){
    //             lineGroup.clearLayers();
    //             commodity_shipped_table('commo_shipped_table', 'service_table', key);
    //             ldc_allocated.forEach((ec_info) => {
    //                 const ec_name = ec_info[0];
    //                 const line = [
    //                     LDC_name_coordinates[key],
    //                     EC_name_coordinates[ec_name],
    //                 ];
    //                 L.polyline(line, {color: 'black'}).addTo(lineGroup);
    //             });
    //         });
    //     });
    //     Object.entries(EC_name_coordinates).forEach(([key, value]) => {
    //         let length = ec_demands[key][0].length;
    //         let demands = new Array(length).fill(0);
    //         for(let i = 0; i < ec_demands[key].length; i++){
    //             for(let j = 0; j < length; j++){
    //                 demands[j] += ec_demands[key][i][j];
    //             }
    //         }
    //         const ec_marker = L.marker([value[0], value[1]])
    //             .addTo(markerGroup)
    //             .bindPopup(
    //                 '<div class="lrt"><br><b>نام مرکز تخلیه: </b>' +
    //                     key + '<br>' +
    //                     '<br>مقدار آب ارسال شده: ' +
    //                     demands[0].toFixed(2) +
    //                     '<br>مقدار غذای ارسال شده: ' +
    //                     demands[1].toFixed(2) +
    //                     '<br>مقدار کیت پزشکی ارسال شده: ' +
    //                     demands[2].toFixed(2) +
    //                     '<br>مقدار چادر ارسال شده: ' +
    //                     demands[3].toFixed(2) + '<br>' +
    //                     '<br>مجموع کالا‌های ارسالی: ' +
    //                     demands.reduce((a, b) => a + b, 0).toFixed(2) + '</div>'
    //             )
    //             .setIcon(redSquare);
    //         ec_marker.on('click', function(){
    //             lineGroup.clearLayers();
    //             const $service_table = $('#service_table');
    //             const $commodity_shipped_table = $('#commo_shipped_table tbody');
    //             $('#commo_shipped_table').removeClass('d-none').addClass('d-table');
    //             $commodity_shipped_table.html('');
    //             for(let i = 1; i < $service_table.find('tr').length; i++){
    //                 const $row = $service_table.find('tr').eq(i);
    //                 if($row.find('td').eq(1).text() === key){
    //                     const $newRow = $('<tr>').on('click', handleRowClick);
    //                     $row.find('td').each(function(){
    //                         $newRow.append($('<td>').html($(this).html()));
    //                     });
    //                     $commodity_shipped_table.append($newRow);
    //                 }
    //             }
    //         });
    //         ec_marker.on('mouseover', function(){
    //             this.openPopup();
    //         });
    //         ec_marker.on('mouseout', function(){
    //             this.closePopup();
    //         });
    //     });
    // }

    // function filling_table(data, table_name){
    //     const $tbody = $('#' + table_name + ' tbody');
    //     $tbody.html('');
    //     Object.entries(data).forEach(([key, value]) => {
    //         value.forEach((point_info) => {
    //             const $newRow = $('<tr>').addClass('row_click');
    //             $newRow.append($('<td>').text(key));
    //             $newRow.append($('<td>').text(point_info[0]));
    //             point_info[1].forEach((commodities_info) => {
    //                 $newRow.append($('<td>').text(commodities_info.toFixed(2)));
    //             });
    //             $newRow.append($('<td>').text(point_info[1].reduce((a, b) => a + b, 0).toFixed(2)));
    //             $newRow.on('click', handleRowClick);
    //             $tbody.append($newRow);
    //         });
    //     });
    // }

    // function commodity_shipped_table(newtable, oldtable, pointname){
    //     const $table1 = $('#' + oldtable);
    //     const $table2 = $('#' + newtable + ' tbody');
    //     $('#' + newtable).removeClass('d-none').addClass('d-table');
    //     $table2.html('');
    //     $table1.find('tr').slice(1).each(function(){
    //         const $row = $(this);
    //         if($row.find('td').eq(0).text() === pointname){
    //             const $newRow = $('<tr>').addClass('row_click');
    //             $row.find('td').each(function(){
    //                 $newRow.append($('<td>').html($(this).html()));
    //             });
    //             $newRow.on('click', handleRowClick);
    //             $table2.append($newRow);
    //         }
    //     });
    // }

    // function filling_additional_unmetdemand_table(data, table_name){
    //     const $table = $('#' + table_name);
    //     $table.find('tbody').html('');
    //     Object.entries(data).forEach(([key, value]) => {
    //         const $newRow = $('<tr>');
    //         $newRow.append($('<td>').text(key));
    //         value.forEach((commo) => {
    //             $newRow.append($('<td>').text(commo.toFixed(2)));
    //         });
    //         $newRow.append($('<td>').text(value.reduce((a, b) => a + b, 0).toFixed(2)));
    //         $table.find('tbody').append($newRow);
    //     });
    // }

    // function handleRowClick(){
    //     const $targetElement = $('#map');
    //     $targetElement[0].scrollIntoView({
    //         behavior: 'smooth',
    //         block: 'start'
    //     });
    //     const $row = $(this);
    //     map.flyTo(Nodes_coordinate[$row.find('td').eq(0).text()], 15);
    //     const line = [
    //         Nodes_coordinate[$row.find('td').eq(0).text()],
    //         Nodes_coordinate[$row.find('td').eq(1).text()],
    //     ];
    //     lineGroup.clearLayers();
    //     L.polyline(line, {color: 'black'}).addTo(lineGroup);
    // }
});

// function filterInventoryDecisionTable(event){
//     const input = event.target;
//     const filter = input.value.toLowerCase();
//     const $table = $('#inventory_table');
//     const $rows = $table.find('tr');
//     let column;
//     if(input.id == 'CMDsearchInput'){
//         column = 0;
//     } else if(input.id == 'LDCsearchInput'){
//         column = 1;
//     }
//     $rows.slice(1).each(function(){
//         const $td = $(this).find('td').eq(column);
//         if($td.length){
//             const txtValue = $td.text() || $td.html();
//             if(txtValue.toLowerCase().includes(filter)){
//                 $(this).show();
//             } else {
//                 $(this).hide();
//             }
//         }
//     });
// }

// function filterServiceDecisionTable(event){
//     const input = event.target;
//     const filter = input.value.toLowerCase();
//     const $table = $('#service_table');
//     const $rows = $table.find('tr');
//     let column;
//     if(input.id == 'LDCsearchInput'){
//         column = 0;
//     } else if(input.id == 'ECsearchInput'){
//         column = 1;
//     }
//     $rows.slice(1).each(function(){
//         const $td = $(this).find('td').eq(column);
//         if($td.length){
//             const txtValue = $td.text() || $td.html();
//             if(txtValue.toLowerCase().includes(filter)){
//                 $(this).show();
//             } else {
//                 $(this).hide();
//             }
//         }
//     });
// }
