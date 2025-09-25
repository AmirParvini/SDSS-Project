import {
    markerGroup,
    idc_icon,
    tmc_icon,
    ec_icon,
    h_icon,
    da_icon,
} from "./map_init.js";

class ShowPoints {
    constructor() {
        // Define icon mapping for each point type
        this.elementVisibility = {
            IDC: {
                demand: true,
                injured: true,
                h_tmc_capacity: true,
                idc_capacity: false,
                cost: false,
            },
            EC: {
                demand: false,
                injured: true,
                h_tmc_capacity: true,
                idc_capacity: true,
                cost: true,
            },
            TMC: {
                demand: true,
                injured: true,
                h_tmc_capacity: false,
                idc_capacity: true,
                cost: false,
            },
            H: {
                demand: true,
                injured: true,
                h_tmc_capacity: false,
                idc_capacity: true,
                cost: true,
            },
            DA: {
                demand: true,
                injured: false,
                h_tmc_capacity: true,
                idc_capacity: true,
                cost: true,
            },
        };

        this.iconMap = {
            IDC: idc_icon,
            EC: ec_icon,
            TMC: tmc_icon,
            H: h_icon,
            DA: da_icon,
        };

        this.params = {
            IDC: {
                capacity: "#idc_capacityEdit",
                fixed_cost: "#pointcostEdit",
            },
            EC: { demand: "#pointdemandEdit" },
            TMC: {
                capacity: "#h_tmc_capacityEdit",
                fixed_cost: "#pointcostEdit",
            },
            H: { capacity: "#h_tmc_capacityEdit" },
            DA: { injured: "#pointinjuredEdit" },
        };
        // Define Persian names for point types
        this.typeNames = {
            IDC: "مرکز توزیع اقلام امدادی",
            EC: "پناهگاه",
            DA: "منطقه آسیب دیده",
            TMC: "مرکز درمانی موقت",
            H: "بیمارستان",
        };
        this.editmodal = new bootstrap.Modal($("#editPointModal"));
        this.selectmarkerid = null;
        this.selectmarkertype = null;
        this.markersById = {};
        this.self = this;
    }

    loadpoints(data) {
        const self = this.self;
        // Clear existing markers
        markerGroup.clearLayers();
        // Iterate through each point type
        Object.entries(data).forEach(([pointType, points]) => {
            const icon = this.iconMap[pointType];
            const typeName = this.typeNames[pointType];
            if (points && Array.isArray(points)) {
                // Add each point to the map
                points.forEach((point) => {
                    if (point.lat && point.lng) {
                        const marker = L.marker([point.lat, point.lng], {
                            icon: icon,
                        }).addTo(markerGroup);
                        marker.id = point.id;
                        self.markersById[point.id] = marker;
                        marker.on("click", function (e) {
                            self.editmodal.show();
                            self.selectmarkerid = point.id;
                            self.selectmarkertype = pointType;
                            $("#pointType .form-control").val(
                                self.typeNames[pointType]
                            );
                            $("#pointNameEdit .form-control").val(point.name);
                            $("#pointLatEdit .form-control").val(point.lat);
                            $("#pointLngEdit .form-control").val(point.lng);
                            $(".showinput").hide();
                            Object.entries(self.params[pointType]).forEach(
                                ([param, elementid]) => {
                                    $("#pointType").show();
                                    $("#pointNameEdit").show();
                                    $("#pointLatEdit").show();
                                    $("#pointLngEdit").show();
                                    $(elementid).show();
                                    console.log(param);
                                    $(`${elementid} .form-control`).val(
                                        point[param]
                                    );
                                }
                            );
                        });
                        // Create popup content with point information
                        let popupContent = `<div dir="rtl" class="text-center">`;
                        popupContent += `<b>${typeName}</b><br>`;
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
                    }
                });
            }
        });
    }
}

export default ShowPoints;
