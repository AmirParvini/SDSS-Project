// ui/ScenarioSelector.js
// -----------------------------------------------------------------------------
// نوارِ سناریو: یک dropdown از سناریوها + دو دکمه‌ی «ایجاد» و «ویرایش».
//
// نگاشت اصول SOLID:
//  - SRP: فقط DOMِ نوار سناریو را می‌شناسد. نه API صدا می‌زند، نه با نقشه/جدول کار دارد.
//  - DIP/Observer: با بیرون فقط از طریق callback حرف می‌زند
//        (onSelect / onCreateClick / onEditClick)؛ تصمیم‌گیری با Controller است.
//  - OCP: این View خودش markupـش را می‌سازد؛ افزودن سناریو نیازی به تغییر این کلاس ندارد.
//
// HTML لازم (فقط یک نقطه‌ی اتصال):  <div id="scenarioBar"></div>
// -----------------------------------------------------------------------------
export default class ScenarioSelector {
    constructor(containerSelector = "#scenarioBar") {
        this._containerSelector = containerSelector;
        this.onSelect = null; // (scenarioId) => void
        this.onCreateClick = null; // () => void
        this.onEditClick = null; // () => void
        this._scenarios = [];
    }

    // ساخت markup، cache گرفتن المان‌ها و اتصال رویدادها.
    init() {
        this._container = $(this._containerSelector);
        // this._render();
        this._cacheElements();
        this._wireEvents();
    }

    _render() {
        this._container.html(`
            <div class="scenario-bar d-flex align-items-center gap-2">
                <select id="scenarioSelect" class="form-select scenario-select"></select>
                <button id="createScenarioBtn" type="button" class="btn btn-primary">ایجاد سناریو</button>
                <button id="editScenarioBtn" type="button" class="btn btn-secondary">ویرایش سناریو</button>
            </div>
        `);
    }

    _cacheElements() {
        this._select = this._container.find("#scenarioSelect");
        this._createBtn = this._container.find("#createScenarioBtn");
        this._editBtn = this._container.find("#editScenarioBtn");
    }

    _wireEvents() {
        this._select.on("change", () => {
            if (this.onSelect) this.onSelect(this._select.val());
        });
        this._createBtn.on("click", () => {
            if (this.onCreateClick) this.onCreateClick();
        });
        this._editBtn.on("click", () => {
            if (this.onEditClick) this.onEditClick();
        });
    }

    // پرکردن dropdown و انتخابِ پیش‌فرضِ سناریوی فعال.
    // scenarios: [{ id, name, description }], activeId: شناسه‌ی فعال در دیتابیس.
    render(scenarios, activeId) {
        this._scenarios = scenarios || [];
        this._select.empty();
        this._scenarios.forEach((s) => {
            const option = $("<option></option>").val(String(s.id)).text(s.name);
            this._select.append(option);
        });

        if (activeId !== undefined && activeId !== null) {
            this._select.val(String(activeId));
        }
    }

    getSelectedId() {
        return this._select.val();
    }

    setSelected(id) {
        this._select.val(String(id));
    }

    // یافتن آبجکتِ کامل سناریو بر اساس id (برای پرکردن مودالِ ویرایش).
    getScenarioById(id) {
        return (
            this._scenarios.find((s) => String(s.id) === String(id)) || null
        );
    }
}
