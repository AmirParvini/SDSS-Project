document.addEventListener('DOMContentLoaded', function(){

    // انتخاب آیتم‌ها و تغییر متن دکمه
    document.querySelectorAll(".dropdown-item").forEach((item) => {
        item.addEventListener("click", function () {
            const dropdownButton =
                this.closest(".btn-group").querySelector(".btn-secondary");
            dropdownButton.textContent = this.textContent;
        });
    });

    let controller;
    let isFetching = false;

    document.getElementById("run").addEventListener("click", function () {
        document.getElementById("commo_shipped_table").classList.add("d-none");
        let district = document.getElementById("district").textContent;
        let APR = document.getElementById("APR").value;
        let PP = document.getElementById("PP").textContent;
        let Config = document.getElementById("Config").textContent;
        if (isFetching == false) {
            isFetching = true;
            this.innerHTML = "stop";
            document.getElementById("error_message").innerHTML = "";
            controller = new AbortController();
            const signal = controller.signal;
            fetch("/solving", {
                signal,
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRF-TOKEN": document
                        .querySelector('meta[name="csrf-token"]')
                        .getAttribute("content"),
                },
                body: JSON.stringify({
                    District: district,
                    APR: APR,
                    PP: PP,
                    Config: Config,
                }),
            })
                .then((response) => {
                    if (!response.ok) {
                        // throw new Error('Network response was not ok');/
                        console.log('error:', response);
                    }
                    return response.json();
                })
                .then((data) => {
                    if (data.status == "success") {
                        console.log(data);
                        const keys = Object.keys(data.output);
                        const values = Object.values(data.output);
                        console.log("keys: ", keys);
                        console.log("values: ", values);
                        FetchedDataParse(values);
                    } else if (data.status == "error") {
                        console.log("error:", data);
                        let errorMsg = data.message || "An error occurred";
                        // if (data.error) {
                        //     errorMsg += `: ${data.error}`;
                        // }
                        document.getElementById("error_message").innerHTML =
                            errorMsg;
                    }
                    this.innerHTML = "Solve Model";
                    isFetching = false;
                })
                .catch((error) => {
                    if (error.name === "AbortError") {
                        document.getElementById("error_message").innerHTML =
                            "Fetch request was canceled.";
                    } else {
                        document.getElementById("error_message").innerHTML =
                            "Fetch failed.";
                        console.error("Fetch error:", error);
                    }
                    this.innerHTML = "Solve Model";
                    isFetching = false;
                });
        } else {
            isFetching = false;
            controller.abort();
            this.innerHTML = "Solve Model";
        }
    });

    var ec_demands = {};
    var ldc_inventories = {};
    var inventory_report_values = {};
    var service_report_values = {};
    function FetchedDataParse(values) {
        let solution_report_id_array = [
            "Solution Status",
            "Iterations",
            "Solution Time (sec)",
            "Total Distance",
            "Total LDC Used",
            "Total Unmet Demand Amount",
            "Total Additional Inventory",
        ];
        const solution_report_values = values[0]; // python output Key: "SolutionInformation"
        inventory_report_values = values[1]; // python output Key: "CMDs_to_LDCs_Allocation"
        service_report_values = values[2]; // python output Key: "LDCs_to_ECs_Allocation"
        const nodes_coordinate_report_values = values[3]; // python output Key: "NodesCoordinate"
        const additional_unmetdemand_report_values = values[4];
        const ec_name_unmetdemand = Object(additional_unmetdemand_report_values.EC_name_unmetdemand)
        const ldc_name_additional_inventory = Object(additional_unmetdemand_report_values.LDC_name_additional_inventory)
        show_nodes(nodes_coordinate_report_values);
        // filling Solution Information table
        Object.entries(solution_report_values).forEach(([key, value], index) => {
            const report_row = document.getElementById(
                solution_report_id_array[index]
            );
            report_row.innerHTML = "";
            report_row.innerHTML = value;
        });
        // filling inventory decision table
        filling_table(inventory_report_values, "inventory_table");
        // filling service decision table
        filling_table(service_report_values, "service_table");
        // filling additional inventory table
        filling_additional_unmetdemand_table(ldc_name_additional_inventory, "additional_inventory_table")
        // filling unmet demand table
        filling_additional_unmetdemand_table(ec_name_unmetdemand, "unmet_demand_table")

    }

    var CMD_name_coordinates = {}
    var LDC_name_coordinates = {}
    var EC_name_coordinates = {}
    var Nodes_coordinate = {}

    function show_nodes(nodes_coordinate_report_values) {
        markerGroup.clearLayers();
        lineGroup.clearLayers();
        const CMD_coordinate = [];
        const LDC_coordinate = [];
        const EC_coordinate = [];
        CMD_name_coordinates = nodes_coordinate_report_values.CMDs_coordinate;
        LDC_name_coordinates = nodes_coordinate_report_values.LDCs_coordinate;
        EC_name_coordinates = nodes_coordinate_report_values.ECs_coordinate;
        Nodes_coordinate = {...CMD_name_coordinates, ...LDC_name_coordinates, ...EC_name_coordinates}
        Object.entries(CMD_name_coordinates).forEach(([key, value]) => {
            const cmd_allocated = inventory_report_values[key];
            cmd_allocated.forEach((ldc_info) => {
                const ldc_name = ldc_info[0];
                const ldc_inventory = ldc_info[1];
                ldc_inventories[ldc_name] = ldc_inventory;
            });
            const cmd_marker = L.marker([value[0], value[1]])
                .addTo(markerGroup)
                .bindPopup("<b>نام پایگاه امداد: </b>" + key);
            cmd_marker.on("mouseover", function () {
                this.openPopup();
            });
            cmd_marker.on("mouseout", function () {
                this.closePopup();
            });
            cmd_marker.addEventListener("click", function () {
                lineGroup.clearLayers();
                commodity_shipped_table(
                    "commo_shipped_table",
                    "inventory_table",
                    key
                );
                cmd_allocated.forEach((ldc_info) => {
                    const ldc_name = ldc_info[0];
                    const line = [
                        CMD_name_coordinates[key],
                        LDC_name_coordinates[ldc_name],
                    ];
                    L.polyline(line, { color: "black" }).addTo(lineGroup);
                });
            });
        });
        Object.entries(LDC_name_coordinates).forEach(([key, value]) => {
            const ldc_allocated = service_report_values[key];
            ldc_allocated.forEach((ec_info) => {
                const ec_name = ec_info[0];
                const ec_demand = ec_info[1];
                if (!ec_demands[ec_name]) {
                    ec_demands[ec_name] = [];
                }
                ec_demands[ec_name].push(ec_demand);
            });
            const ldc_marker = L.marker([value[0], value[1]])
                .addTo(markerGroup)
                .bindPopup(
                    "<div class='lrt'><br><b>نام مرکز توزیع محلی: </b>" +
                        key + "<br>" +
                        "<br>مقدار موجودی آب: " +
                        ldc_inventories[key][0].toFixed(2) +
                        "<br>مقدار موجودی غذا: " +
                        ldc_inventories[key][1].toFixed(2) +
                        "<br>مقدار موجودی کیت پزشکی: " +
                        ldc_inventories[key][2].toFixed(2) +
                        "<br>مقدار موجودی چادر امداد: " +
                        ldc_inventories[key][3].toFixed(2) + "<br>" +
                        "<br>مجموع موجودی کالا‌ها: " +
                        ldc_inventories[key].reduce((a, b) => a + b, 0).toFixed(2) + '</div>'
                )
                .setIcon(blueCircle);
            ldc_marker.on("mouseover", function () {
                this.openPopup();
            });
            ldc_marker.on("mouseout", function () {
                this.closePopup();
            });
            ldc_marker.addEventListener("click", function () {
                lineGroup.clearLayers();
                commodity_shipped_table(
                    "commo_shipped_table",
                    "service_table",
                    key
                );
                ldc_allocated.forEach((ec_info) => {
                    const ec_name = ec_info[0];
                    const line = [
                        LDC_name_coordinates[key],
                        EC_name_coordinates[ec_name],
                    ];
                    L.polyline(line, { color: "black" }).addTo(lineGroup);
                });
            });
        });
        Object.entries(EC_name_coordinates).forEach(([key, value]) => {
            let length = ec_demands[key][0].length;
            let demands = new Array(length).fill(0); // [0, 0, 0]
            for (let i = 0; i < ec_demands[key].length; i++) {
                for (let j = 0; j < length; j++) {
                    demands[j] += ec_demands[key][i][j];
                }
            }
            const ec_marker = L.marker([value[0], value[1]])
                .addTo(markerGroup)
                .bindPopup(
                    "<div class='lrt'><br><b>نام مرکز تخلیه: </b>" +
                        key + "<br>" +
                        "<br>مقدار آب ارسال شده: " +
                        demands[0].toFixed(2) +
                        "<br>مقدار غذای ارسال شده: " +
                        demands[1].toFixed(2) +
                        "<br>مقدار کیت پزشکی ارسال شده: " +
                        demands[2].toFixed(2) +
                        "<br>مقدار چادر ارسال شده: " +
                        demands[3].toFixed(2) + "<br>" +
                        "<br>مجموع کالا‌های ارسالی: " +
                        demands.reduce((a, b) => a + b, 0).toFixed(2) + "</div>"
                )
                .setIcon(redSquare);
            ec_marker.addEventListener("click", function () {
                lineGroup.clearLayers();
                const service_table = document.getElementById("service_table");
                const commodity_shipped_table = document
                    .getElementById("commo_shipped_table")
                    .getElementsByTagName("tbody")[0];
                document
                    .getElementById("commo_shipped_table")
                    .classList.remove("d-none");
                document
                    .getElementById("commo_shipped_table")
                    .classList.add("d-table");
                commodity_shipped_table.innerHTML = "";
                for (var i = 1; i < service_table.rows.length; i++) {
                    if (service_table.rows[i].cells[1].textContent === key) {
                        var newRow = commodity_shipped_table.insertRow();
                        newRow.addEventListener('click', handleRowClick);
                        for (
                            var j = 0;
                            j < service_table.rows[i].cells.length;
                            j++
                        ) {
                            var newCell = newRow.insertCell();
                            newCell.textContent =
                                service_table.rows[i].cells[j].innerHTML;
                        }
                        const line = [
                            EC_name_coordinates[key],
                            LDC_name_coordinates[
                                service_table.rows[i].cells[0].textContent
                            ],
                        ];
                        L.polyline(line, { color: "black" }).addTo(lineGroup);
                    }
                }
            });
            ec_marker.on("mouseover", function () {
                this.openPopup();
            });
            ec_marker.on("mouseout", function () {
                this.closePopup();
            });
        });
        // map.flyTo([35.6892, 51.389], 13, {
        //     animate: true,
        //     duration: 1, // مدت زمان انیمیشن بر حسب ثانیه
        // });
    }

    function filling_table(data, table_name) {
        const table = document
            .getElementById(table_name)
            .getElementsByTagName("tbody")[0];
        table.innerHTML = "";
        Object.entries(data).forEach(([key, value]) => {
            value.forEach((point_info) => {
                newRow = table.insertRow();
                newRow.classList.add("row_click");
                newRow.insertCell().textContent = key;
                newRow.insertCell().textContent = point_info[0];
                point_info[1].forEach((commodities_info) => {
                    newRow.insertCell().textContent = commodities_info.toFixed(2);
                });
                newRow.insertCell().textContent = point_info[1].reduce((a, b) => a + b, 0).toFixed(2)
                newRow.addEventListener('click', handleRowClick);
            });
        });
    }

    // جدول پایین نقشه که با هر کلیک روی نقطه نمایش داده میشه
    function commodity_shipped_table(newtable, oldtable, pointname) {
        var table1 = document.getElementById(oldtable);
        var table2 = document
            .getElementById(newtable)
            .getElementsByTagName("tbody")[0];
        document.getElementById(newtable).classList.remove("d-none");
        document.getElementById(newtable).classList.add("d-table");
        table2.innerHTML = "";
        // از ردیف دوم (index=1) به بعد پیمایش می‌کنیم (ردیف اول عنوان است)
        for (var i = 1; i < table1.rows.length; i++) {
            // console.log(pointname)
            if (table1.rows[i].cells[0].textContent === pointname) {
                // ستون دوم: نام
                var newRow = table2.insertRow();
                newRow.classList.add("row_click");
                for (var j = 0; j < table1.rows[i].cells.length; j++) {
                    var newCell = newRow.insertCell();
                    newCell.textContent = table1.rows[i].cells[j].innerHTML;
                }
                newRow.addEventListener('click', handleRowClick);
            }
        }
    }

    function filling_additional_unmetdemand_table(data, table_name){
        const table = document.getElementById(table_name);
        table.getElementsByTagName('tbody')[0] = '';
        Object.entries(data).forEach(([key, value]) => {
            var newRow = table.insertRow();
            newRow.insertCell().textContent = key;
            value.forEach((commo) => {
                newRow.insertCell().textContent = commo.toFixed(2);
            });
            newRow.insertCell().textContent = value.reduce((a, b) => a + b, 0).toFixed(2)
        });
    }



    // تعریف تابع برای هر کلیک
    function handleRowClick() {
        const targetElement = document.getElementById('map');
        targetElement.scrollIntoView({
            behavior: 'smooth', // برای اسکرول نرم و انیمیشنی
            block: 'start'      // المنت در بالای viewport قرار می‌گیرد
            // block: 'center'  // المنت در وسط viewport قرار می‌گیرد (اختیاری)
        });
        // 'this' به عنصری اشاره می‌کند که رویداد روی آن رخ داده (یعنی سطر 'tr')
        const row = this;
        map.flyTo(Nodes_coordinate[row.cells[0].textContent], 15)
        const line = [
            Nodes_coordinate[row.cells[0].textContent],
            Nodes_coordinate[
                row.cells[1].textContent
            ],
        ];
        lineGroup.clearLayers()
        L.polyline(line, { color: "black" }).addTo(lineGroup);
    }




    var map = L.map("map").setView([35.7, 51.39], 11);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution:
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    let lineGroup = L.layerGroup().addTo(map);
    let markerGroup = L.layerGroup().addTo(map);
    var blueCircle = L.divIcon({
        className: "custom-icon",
        html: '<div style="width: 10px; height: 10px; background-color: blue; border-radius: 50%;"></div>',
    });
    var redSquare = L.divIcon({
        className: "custom-icon", // کلاس CSS سفارشی
        html: '<div style="width: 6px; height: 6px; background-color: red; border-radius: 50%;"></div>',
    });
});


function filterInventoryDecisionTable(event) {
    const input = event.target;
    const filter = input.value.toLowerCase();
    const table = document.getElementById("inventory_table");
    const rows = table.getElementsByTagName("tr");
    if (input.id == "CMDsearchInput") {
        var column = 0;
    } else if (input.id == "LDCsearchInput") {
        var column = 1;
    }
    // از سطر دوم به بعد (رد کردن header)
    for (let i = 1; i < rows.length; i++) {
        const td = rows[i].getElementsByTagName("td")[column]; // فقط ستون اول
        if (td) {
            const txtValue = td.textContent || td.innerText;
            if (txtValue.toLowerCase().includes(filter)) {
                rows[i].style.display = ""; // نمایش
            } else {
                rows[i].style.display = "none"; // پنهان‌سازی
            }
        }
    }
}

function filterServiceDecisionTable(event) {
    const input = event.target;
    const filter = input.value.toLowerCase();
    const table = document.getElementById("service_table");
    const rows = table.getElementsByTagName("tr");
    if (input.id == "LDCsearchInput") {
        var column = 0;
    } else if (input.id == "ECsearchInput") {
        var column = 1;
    }
    // از سطر دوم به بعد (رد کردن header)
    for (let i = 1; i < rows.length; i++) {
        const td = rows[i].getElementsByTagName("td")[column]; // فقط ستون اول
        if (td) {
            const txtValue = td.textContent || td.innerText;
            if (txtValue.toLowerCase().includes(filter)) {
                rows[i].style.display = ""; // نمایش
            } else {
                rows[i].style.display = "none"; // پنهان‌سازی
            }
        }
    }
}