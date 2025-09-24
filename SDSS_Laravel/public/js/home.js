import {
    map,
    lineGroup,
    markerGroup,
    blueCircle,
    redSquare,
} from "./map_init.js";

$(document).ready(function () {
    // Map click event to add point
    map.on("click", function (e) {
        $("#pointLat").val(e.latlng.lat);
        $("#pointLng").val(e.latlng.lng);
        $("#addPointModal").modal("show");
    });

    const elementVisibility = {
        IDC: { demand: true, injured: true, h_tmc_capacity: true, idc_capacity: false, cost: false },
        EC: { demand: false, injured: true, h_tmc_capacity: true, idc_capacity: true, cost: true },
        TMC: { demand: true, injured: true, h_tmc_capacity: false, idc_capacity: true, cost: false },
        H: { demand: true, injured: true, h_tmc_capacity: false, idc_capacity: true, cost: true },
        DA: { demand: true, injured: false, h_tmc_capacity: true, idc_capacity: true, cost: true }
    };
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
            .then((response) => response.json())
            .then((data) => {
                if (data.success) {
                    // Add marker to map
                    const lat = $("#pointLat").val();
                    const lng = $("#pointLng").val();
                    const name = $("#pointName").val();
                    L.marker([lat, lng])
                        .addTo(markerGroup)
                        .bindPopup("<b>" + name + "</b>");
                    $("#addPointModal").modal("hide");
                    $("#addPointForm")[0].reset();
                    // console.log(typeof(lat));
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
