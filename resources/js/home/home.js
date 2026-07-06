// home.js  (نقطه‌ی ورود / Composition Root + Controller)
// -----------------------------------------------------------------------------
// این فایل فقط «هماهنگ‌کننده» است: ماژول‌ها را می‌سازد، به هم وصل می‌کند و
// جریان کار (use-case)ها را هدایت می‌کند. هیچ منطق DOM یا نقشه‌ای اینجا نیست.
//
// نگاشت اصول SOLID در این ماژول:
//  - SRP: هر کلاس یک مسئولیت. Controller فقط ارکستراسیون.
//  - OCP: نوع نقطه‌ی جدید = فقط یک ورودی در config/nodeConfig.js.
//  - LSP: همه‌ی Viewها اینترفیس callback یکنواخت دارند.
//  - ISP: هر View فقط متدهایی را که لازم دارد بیرون می‌دهد.
//  - DIP: Controller به وابستگی‌ها از طریق constructor injection وصل است،
//         نه به پیاده‌سازی‌های مشخص (axios/DOM/Leaflet مستقیم).
// -----------------------------------------------------------------------------
import NodeMapView from "./map/NodeMapView.js";
import NodeApiService from "./services/NodeApiService.js";
import AddPointModal from "./ui/AddPointModal.js";
import NodePropertyPanel from "./ui/NodePropertyPanel.js";
import NodeTableView from "./ui/NodeTableView.js";

class HomeController {
    constructor({ api, mapView, tableView, panel, addPointModal }) {
        this.api = api;
        this.mapView = mapView;
        this.tableView = tableView;
        this.panel = panel;
        this.addPointModal = addPointModal;
        this.hscParameters = null;
    }

    init() {
        this._wireEvents();
        this._loadData();
    }

    // اتصال همه‌ی رویدادها در یک نقطه‌ی متمرکز (خواناتر از پخش‌شدن در کل فایل).
    _wireEvents() {
        // جدول: منبع لایه‌ها را از نقشه می‌گیرد و کلیک سطر را گزارش می‌دهد.
        this.tableView.setLayerProvider((type) => this.mapView.getLayers(type));
        this.tableView.bindToggle();
        this.tableView.onRowClick(({ type, id, latlng }) => {
            this.mapView.focus(latlng);
            this.mapView.selectLayerById(type, id);
        });

        // دکمه افزودن نقطه
        $("#addPointBtn").on("click", () => {
            this.addPointModal.enterAddMode();
            this.mapView.setCursor("crosshair");
        });

        // نقشه: کلیک روی نود → فعال‌سازی پنل و نمایش پراپرتی‌ها.
        this.mapView.onNodeClick(({ type, latlng, props }) => {
            this.panel.activateActions();
            this.mapView.showPulse(latlng);
            this.panel.setLatLng(latlng);
            this.panel.showType(type);
            this.panel.displayProps(type, props);
        });

        // نقشه: کلیک روی نقشه → باز کردن modal افزودن نقطه (در حالت add mode).
        this.mapView.onMapClick(({ latlng }) => {
            if (this.addPointModal.isInAddMode()) {
                this.addPointModal.show(latlng);
                this.mapView.resetCursor();
            }
        });

        // پنل: ذخیره/حذف را به use-caseهای Controller واگذار می‌کند.
        this.panel.bind();
        this.panel.onUpdate = (data) => this._updateNode(data);
        this.panel.onDelete = (id, type) => this._deleteNode(id, type);

         // Wire add point modal save handler
        this.addPointModal.onCreate = (data) => this._createNode(data);
    }

    // ---- use-case: بارگذاری اولیه ------------------------------------------
    _loadData() {
        this.api
            .loadData()
            .then(({ geojsonNodes, hscParameters }) => {
                this.hscParameters = hscParameters;
                this.mapView.renderNodes(geojsonNodes);
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while loading data.");
            });
    }

    // ---- use-case: ذخیره (ایجاد/ویرایش) -----------------------------------
    _updateNode(data) {
        this.api
            .updateNode(data)
            .then((res) => {
                if (!res.success) {
                    alert("Error: " + res.message);
                    return;
                }
                alert("Saved successfully!");
                const props = this.mapView.updateNode(data.id, data.type, data);
                if (props) this.panel.displayProps(data.type, props);
                this.tableView.populate(data.type);
                this.panel.exitEditMode();
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while saving.");
            });
    }

    _createNode(data) {
        this.api
            .createNode(data)
            .then((res) => {
                if (!res.success) {
                    alert("Error: " + res.message);
                    return;
                }
                alert("Saved successfully!");
                let marker = this.mapView.createNode(res["node_id"], data);
                this.mapView.showPulse(marker.getLatLng());
                this.panel.displayProps(data.type, marker.options.properties);
                this.tableView.populate(data.type);
                this.panel.activateActions();
                this.panel.exitEditMode();
                this.addPointModal.hide();
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while saving.");
            });
    }

    // ---- use-case: حذف ------------------------------------------------------
    _deleteNode(id, type) {
        if (!id) {
            alert("No node selected for deletion.");
            return;
        }
        if (!confirm("Are you sure you want to delete this node?")) return;

        this.api
            .deleteNode(id)
            .then((res) => {
                if (!res.success) {
                    alert("Error: " + res.message);
                    return;
                }
                alert("Deleted successfully!");
                this.mapView.removeNode(id, type);
                this.tableView.populate(type);
                this.panel.reset();
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while deleting.");
            });
    }
}

// ---- Composition Root: تنها جایی که پیاده‌سازی‌های واقعی به هم گره می‌خورند ---
$(function () {
    const addPointModal = new AddPointModal();
    addPointModal.init();

    const controller = new HomeController({
        api: new NodeApiService(),
        mapView: new NodeMapView(),
        tableView: new NodeTableView(),
        panel: new NodePropertyPanel(),
        addPointModal: addPointModal,
    });
    controller.init();
});
