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
import { HSC_PARAMETER_DEFAULTS } from "./config/hscParametersConfig.js";
import NodeMapView from "./map/NodeMapView.js";
import ResultPopupBuilder from "./results/ResultPopupBuilder.js";
import ResultSet from "./results/ResultSet.js";
import HscParameterService from "./services/HscParameterService.js";
import NodeApiService from "./services/NodeApiService.js";
import ResultService from "./services/ResultService.js";
import ScenarioService from "./services/ScenarioService.js";
import TopsisService from "./services/TopsisService.js";
import AddPointModal from "./ui/AddPointModal.js";
import AllocationTableView from "./ui/AllocationTableView.js";
import CostsReportView from "./ui/CostsReportView.js";
import HscParameterModal from "./ui/HscParameterModal.js";
import NodePropertyPanel from "./ui/NodePropertyPanel.js";
import NodeTableView from "./ui/NodeTableView.js";
import ParetoTableView from "./ui/ParetoTableView.js";
import ReportLayout from "./ui/ReportLayout.js";
import ScenarioModal from "./ui/ScenarioModal.js";
import ScenarioSelector from "./ui/ScenarioSelector.js";

class HomeController {
    constructor({
        api,
        scenarioApi,
        hscApi,
        resultApi,
        topsisApi,
        mapView,
        tableView,
        panel,
        addPointModal,
        scenarioSelector,
        scenarioModal,
        hscModal,
        paretoView,
        allocationView,
        costsView,
        reportLayout,
        popupBuilder,
    }) {
        this.api = api;
        this.scenarioApi = scenarioApi;
        this.hscApi = hscApi;
        this.resultApi = resultApi;
        this.topsisApi = topsisApi;
        this.mapView = mapView;
        this.tableView = tableView;
        this.panel = panel;
        this.addPointModal = addPointModal;
        this.scenarioSelector = scenarioSelector;
        this.scenarioModal = scenarioModal;
        this.hscModal = hscModal;
        this.hscParameters = null;

        // --- report / results collaborators ---
        this.paretoView = paretoView;
        this.allocationView = allocationView;
        this.costsView = costsView;
        this.reportLayout = reportLayout;
        this.popupBuilder = popupBuilder;
        this.resultSet = null; // ResultSet after the model runs
        this.currentSolution = null; // SolutionModel of the selected row
        this.reportMode = false;
    }

    init() {
        this._wireEvents();
        this._wireScenarioEvents();
        this._wireResultEvents();

        // سناریوها را بارگذاری می‌کند (dropdown با پیش‌فرضِ سناریوی فعال) و
        // داده‌ی سناریوی فعال را روی نقشه می‌آورد. این دو مستقل‌اند.
        this._loadScenarios();
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
        // در حالت گزارش، کلیک روی نود رفتار متفاوتی دارد (پاپ‌آپ + مسیرها).
        this.mapView.onNodeClick((payload) => {
            if (this.reportMode) {
                this._onReportNodeClick(payload);
                return;
            }
            const { type, latlng, props } = payload;
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

        // دکمه‌ی پارامترهای HSC: باز کردن مودال با مقادیرِ سناریوی فعال (فقط‌خواندنی).
        $("#hsc_parameters").on("click", () =>
            this._openHscParameters(this.scenarioSelector.getSelectedId()),
        );
        this.hscModal.onSave = (data) =>
            this._updateHscParameters(
                data,
                this.scenarioSelector.getSelectedId(),
            );
    }

    // اتصال رویدادهای مربوط به سناریو (انتخاب، ایجاد، ویرایش).
    _wireScenarioEvents() {
        // تعویض سناریو از روی dropdown.
        this.scenarioSelector.onSelect = (id) => this._switchScenario(id);

        // دکمه‌ی «ایجاد» → مودالِ خالی.
        this.scenarioSelector.onCreateClick = () =>
            this.scenarioModal.openForCreate();

        // دکمه‌ی «ویرایش» → مودال با اطلاعاتِ سناریوی انتخاب‌شده.
        this.scenarioSelector.onEditClick = () => {
            const id = this.scenarioSelector.getSelectedId();
            const current = this.scenarioSelector.getScenarioById(id);
            if (current) this.scenarioModal.openForEdit(current);
        };

        // ثبتِ مودال (ایجاد یا ویرایش بسته به mode).
        this.scenarioModal.onSubmit = (data, mode) => {
            if (mode === "edit") {
                this._updateScenario(data);
            } else {
                this._createScenario(data);
            }
        };
    }

    // =========================================================================
    // Report / results use-cases
    // =========================================================================

    // اتصال رویدادهای اجرای مدل، تب‌های داشبورد/گزارش و جدول‌های نتایج.
    _wireResultEvents() {
        $("#runModelBtn").on("click", () => this._runModel());
        $("#reportsTab").on("click", (e) => {
            e.preventDefault();
            this._enterReportMode();
        });
        $("#dashboardTab").on("click", (e) => {
            e.preventDefault();
            this._exitReportMode();
        });

        // انتخاب یک جواب از جدول پرتو → فیلتر نقشه + جدول تخصیص + کانتینر هزینه.
        this.paretoView.onRowSelect = (id) => this._selectSolution(id);

        // دکمه‌ی TOPSIS در جدول پرتو → رتبه‌بندی جواب‌های روی صفحه + هایلایت برنده.
        this.paretoView.onRunTopsis = (payload) => this._runTopsis(payload);

        // کلیک روی سطر جدول تخصیص → تمرکز روی مبدأ و نمایش مسیرهای آن.
        this.allocationView.onRowSelect = (payload) =>
            this._focusAllocationRow(payload);

        this.allocationView.nameResolver = (type, id) =>
            this.mapView.getNameById(type, id);
    }

    // ---- use-case: اجرای مدل سمت سرور --------------------------------------
    // هیچ داده‌ای ارسال نمی‌شود؛ فقط یک درخواست به کنترلر solving زده می‌شود و
    // خروجی (لیست جواب‌ها) نگه‌داری می‌شود تا در حالت گزارش نمایش داده شود.
    _runModel() {
        const $btn = $("#runModelBtn");
        $btn.prop("disabled", true).addClass("is-loading");
        $btn.find(".run-model__busy p").text("Checking and calculating routes...");
        $("#reportsTab").addClass("disabled");
        $("#hsc_parameters").parent().addClass("d-none");
        $("#scenarioBar").addClass("d-none");
        $("#createScenarioBtn").addClass("d-none");
        $("#editScenarioBtn").addClass("d-none");
        $("#cancelRunningBtn").parent().removeClass("d-none");
        this.resultApi
            .calculatePaths()
            .then(() => {
                $btn.find(".run-model__busy p").text("Running optimization...");
                return this.resultApi.runModel();
            })
            .then((payload) => {
                this.resultSet = new ResultSet(
                    payload,
                    this.scenarioSelector.getSelectedId(),
                );
                if (this.resultSet.isEmpty()) {
                    alert("The model returned no solutions.");
                    return;
                }
                alert(
                    "Model finished successfully. Open Reports to view the results.",
                );
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while running the model.");
            })
            .finally(() => {
                $btn.prop("disabled", false).removeClass("is-loading");
                $btn.find(".run-model__busy p").text("Running...");
                $("#reportsTab").removeClass("disabled");
                $("#hsc_parameters").parent().removeClass("d-none");
                $("#scenarioBar").removeClass("d-none");
                $("#createScenarioBtn").removeClass("d-none");
                $("#editScenarioBtn").removeClass("d-none");
                $("#cancelRunningBtn").parent().addClass("d-none");
            });
    }

    // ---- use-case: ورود به حالت گزارش --------------------------------------
    _enterReportMode() {
        if (!this.resultSet || this.resultSet.isEmpty()) {
            alert("Please run the model first.");
            return;
        }
        $("#runModelBtn").parent().addClass("d-none");
        $("#hsc_parameters").parent().addClass("d-none");
        $("#createScenarioBtn").addClass("d-none");
        this.reportMode = true;
        this.reportLayout.enterReport();
        this.panel.reset();
        this.mapView.removePulse();

        this.paretoView.render(this.resultSet.toParetoRows());
        this.paretoView.show();

        // اولین جواب به‌صورت پیش‌فرض انتخاب می‌شود تا صفحه خالی نماند.
        const first = this.resultSet.list()[0];
        if (first) {
            this.paretoView.setActive(first.id);
            this._selectSolution(first.id);
        }
    }

    // ---- use-case: خروج از حالت گزارش (دکمه Dashboard) ----------------------
    _exitReportMode() {
        $("#runModelBtn").parent().removeClass("d-none");
        $("#hsc_parameters").parent().removeClass("d-none");
        $("#createScenarioBtn").removeClass("d-none");
        this.reportMode = false;
        this.mapView.closePopups();
        this.reportLayout.enterDashboard();
        this.paretoView.hide();
        this.allocationView.hide();
        this.allocationView.clear();
        this.costsView.hide();
        this.mapView.clearGeometries();
        this.mapView.showAllNodes();
        this.currentSolution = null;
    }

    // ---- use-case: انتخاب یک جواب از جدول پرتو ------------------------------
    // فقط مارکرهای همان جواب روی نقشه می‌مانند، جدول تخصیص و کانتینر هزینه پر می‌شوند.
    _selectSolution(id) {
        const solution = this.resultSet.getSolution(id);
        if (!solution) return;
        this.currentSolution = solution;

        this.mapView.clearGeometries();
        this.mapView.filterToSolution(solution.getNodeIdsByType());

        this.allocationView.setSolution(solution);
        this.allocationView.show();

        this.costsView.render(solution.costs, solution.id);
        this.costsView.show();
    }

    // ---- use-case: رتبه‌بندی جواب‌های پرتو با Topsis -------------------------------
    // شناسه‌ها/مقادیر همان چیزیه‌اند که همین اکنون در جدول دیده می‌شوند
    // (ParetoTableView آن‌ها را می‌خواند؛ هیچ کوئری تازه‌ای از بک‌اند زده نمی‌شود)؛
    // پاسخ فقط solution_id برنده است که برای هایلایت سطر مربوطه به ParetoTableView برمی‌گردد.
    _runTopsis({ solutionIds, values, weights }) {
        this.paretoView.setTopsisLoading(true);
        this.topsisApi
            .getBestSolutionId({ solutionIds, values, weights })
            .then((bestSolutionId) => {
                this.paretoView.highlightBestSolution(bestSolutionId);
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while running Topsis.");
            })
            .finally(() => {
                this.paretoView.setTopsisLoading(false);
            });
    }

    // ---- تمرکز روی یک نود در حالت گزارش -------------------------------------
    // نقشه روی مارکر می‌رود، مسیرهای تخصیصِ آن نود رسم می‌شوند و پاپ‌آپ باز می‌شود.
    // این متد هم برای کلیک سطر جدول تخصیص و هم برای کلیک روی خود مارکر استفاده می‌شود.
    _focusNode(type, id, layer = null, props = {}) {
        if (!this.currentSolution) return;
        const solution = this.currentSolution;

        const latlng = this.mapView.getLatLngById(type, id);
        if (latlng) this.mapView.focus(latlng);

        this.mapView.drawGeometries(solution.getNodeGeometries(type, id));

        const html = this.popupBuilder.build(solution, type, id, props);
        this.mapView.openReportPopup(type, id, html, layer);
    }

    // کلیک روی سطر جدول تخصیص: package_flows روی مبدأ (DC) تمرکز می‌کند؛
    // سه تخصیص دیگر روی target کلیک‌شده و فقط جئومتری‌های همان target.
    _focusAllocationRow({
        allocationType,
        sourceType,
        sourceId,
        targetType,
        targetId,
    }) {
        if (!this.currentSolution) return;
        const solution = this.currentSolution;

        const geometries = solution.getAllocationGeometry(allocationType, sourceId, targetId);

        const type = sourceType;
        const id = sourceId;

        const latlng = this.mapView.getLatLngById(type, id);
        if (latlng) this.mapView.focus(latlng);

        this.mapView.drawGeometries(geometries);

        const html = this.popupBuilder.build(solution, type, id, {});
        this.mapView.openReportPopup(type, id, html);
    }

    // کلیک روی مارکر در حالت گزارش.
    _onReportNodeClick({ type, layer, props }) {
        const id = props ? props.id : null;
        if (id === null || id === undefined) return;
        this._focusNode(type, id, layer, props);
    }

    // ---- use-case: بارگذاری لیست سناریوها ----------------------------------
    _loadScenarios() {
        this.scenarioApi
            .listScenarios()
            .then(({ scenarios, activeId }) => {
                if (!scenarios.length == 0){
                    $("#addPointBtn").removeClass("disabled");

                    // dropdown به‌صورت پیش‌فرض روی سناریوی فعالِ دیتابیس تنظیم می‌شود.
                    this.scenarioSelector.render(scenarios, activeId);
                }
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while loading scenarios.");
            });
    }

    // ---- use-case: تعویض سناریو -------------------------------------------
    // سناریوی قبلی غیرفعال و سناریوی انتخاب‌شده فعال می‌شود، سپس تمام داده‌های
    // فرانت ریست و دوباره از صفر بارگذاری می‌شوند.
    _switchScenario(id) {
        this.scenarioApi
            .activateScenario(id)
            .then((res) => {
                if (!res.success) {
                    alert("Error: " + res.message);
                    return;
                }
                this._resetFrontend();
                this._loadData();
            })
            .then(() => {
                this._loadResults();
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while switching scenario.");
            });
    }

    // ---- use-case: ایجاد سناریو -------------------------------------------
    // مانند رویداد انتخاب سناریو: بک‌اند سناریوی جدید را فعال می‌کند؛ سپس لیست
    // دوباره خوانده می‌شود (تا سناریوی جدید و فعالِ جدید نمایش داده شوند) و همه‌ی
    // داده‌های فرانت ریست می‌شوند تا کاربر داده‌ی تازه وارد کند.
    _createScenario(data) {
        this.scenarioApi
            .createScenario(data)
            .then((res) => {
                if (!res.success) {
                    alert("Error: " + res.message);
                    return Promise.reject("handled");
                }
                $("#addPointBtn").removeClass("disabled");
                this.scenarioModal.close();
                // ذخیره‌ی مقادیر پیش‌فرضِ HSC برای سناریوی تازه‌فعال‌شده.
                return this.hscApi.saveParameters(HSC_PARAMETER_DEFAULTS);
            })
            .then(() => this.scenarioApi.listScenarios())
            .then((list) => {
                this.scenarioSelector.render(list.scenarios, list.activeId);
                this._resetFrontend();
                this._loadData();
            })
            .catch((error) => {
                if (error === "handled") return;
                console.error(error);
                alert("An error occurred while creating scenario.");
            });
    }

    // ---- use-case: پارامترهای HSC -----------------------------------------
    // باز کردن مودال با مقادیرِ سناریوی فعال. ورودی‌ها به‌صورت پیش‌فرض قفل‌اند.
    _openHscParameters(scenario_id) {
        this.hscApi
            .getParameters(scenario_id)
            .then((params) => {
                this.hscModal.display(params);
                this.hscModal.open();
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while loading HSC parameters.");
            });
    }

    // ذخیره‌ی پارامترهای ویرایش‌شده و بازگشتِ مودال به حالت فقط‌خواندنی.
    _updateHscParameters(data, scenario_id) {
        this.hscApi
            .updateParameters(data, scenario_id)
            .then((res) => {
                if (!res.success) {
                    alert("Error: " + res.message);
                    return;
                }
                alert("Saved successfully!");
                this.hscModal.display(data);
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while saving HSC parameters.");
            });
    }

    // ---- use-case: ویرایش سناریو ------------------------------------------
    // فقط name/description تغییر می‌کند؛ سناریوی فعال عوض نمی‌شود، پس داده‌ها ریست نمی‌شوند.
    _updateScenario(data) {
        this.scenarioApi
            .updateScenario(data)
            .then((res) => {
                if (!res.success) {
                    alert("Error: " + res.message);
                    return null;
                }
                this.scenarioModal.close();
                return this.scenarioApi.listScenarios();
            })
            .then((list) => {
                if (!list) return;
                this.scenarioSelector.render(list.scenarios, list.activeId);
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while updating scenario.");
            });
    }

    // ---- ریستِ کاملِ داده‌های فرانت ---------------------------------------
    // پیش از بارگذاری مجددِ داده‌ها فراخوانی می‌شود: نقشه، جدول و پنل پاک می‌شوند.
    _resetFrontend() {
        this.mapView.clearAll();
        this.tableView.reset();
        this.panel.reset();
        this.hscParameters = null;
    }

    // ---- use-case: بارگذاری اولیه ------------------------------------------
    // متد _loadData خودش داده‌های سناریوی اکتیو را از بک‌اند می‌گیرد.
    _loadData() {
        this.api
            .loadData()
            .then(({ geojsonNodes, hscParameters }) => {
                this.hscParameters = hscParameters;
                this.mapView.renderNodes(geojsonNodes);
            })
            .then(() => {
                this._loadResults();
            })
            .catch((error) => {
                console.error(error);
                alert("An error occurred while loading data.");
            });
    }

    _loadResults() {
        let scenario_id = this.scenarioSelector.getSelectedId();
        this.resultApi.fetchReports(scenario_id).then((payload) => {
            this.resultSet = new ResultSet(payload, scenario_id);
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
                this.panel.showType(data.type);
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

    const scenarioSelector = new ScenarioSelector();
    scenarioSelector.init();

    const scenarioModal = new ScenarioModal();
    scenarioModal.init();

    const hscModal = new HscParameterModal();
    hscModal.init();

    // آدرس اجرای مدل از data-attribute دکمه‌ی Run Model خوانده می‌شود تا
    // مسیرِ دقیقِ لاراول استفاده شود (نه یک آدرس حدسی).

    const controller = new HomeController({
        api: new NodeApiService(),
        scenarioApi: new ScenarioService(),
        hscApi: new HscParameterService(),
        resultApi: new ResultService(),
        resultApi: new ResultService(),
        topsisApi: new TopsisService(),
        mapView: new NodeMapView(),
        tableView: new NodeTableView(),
        panel: new NodePropertyPanel(),
        addPointModal: addPointModal,
        scenarioSelector: scenarioSelector,
        scenarioModal: scenarioModal,
        hscModal: hscModal,
        paretoView: new ParetoTableView(),
        allocationView: new AllocationTableView(),
        costsView: new CostsReportView(),
        reportLayout: new ReportLayout(),
        popupBuilder: new ResultPopupBuilder(),
    });
    controller.init();
});
