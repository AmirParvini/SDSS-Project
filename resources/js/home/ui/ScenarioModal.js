// ui/ScenarioModal.js
// -----------------------------------------------------------------------------
// مودالِ «ایجاد/ویرایش سناریو»: فقط دو ویژگی name و description را می‌گیرد.
//
// نگاشت اصول SOLID:
//  - SRP: تنها مسئولیتش DOMِ همین مودال است. عملیات ذخیره (API) اینجا نیست؛
//         مودال فقط داده را جمع می‌کند و رویداد onSubmit را می‌فرستد.
//  - LSP: مثل AddPointModal یک اینترفیس یکنواختِ callback-محور دارد.
//  - DIP/Observer: از طریق onSubmit(data, mode) با Controller حرف می‌زند.
//
// این View خودش markupـش را می‌سازد و به <body> می‌چسباند؛ نیازی به HTML اضافه نیست.
// -----------------------------------------------------------------------------
export default class ScenarioModal {
    constructor() {
        this.onSubmit = null; // (data, mode) => void  |  data: { id?, name, description }
        this._mode = "create"; // "create" | "edit"
        this._editingId = null;
    }

    init() {
        this._render();
        this._cacheElements();
        this._wireEvents();
    }

    _render() {
        if ($("#scenarioModal").length) return; // جلوگیری از ساختِ دوباره
        $("body").append(`
            <div id="scenarioModal" class="scenario-modal" style="display:none; position:fixed; inset:0; z-index:1050; align-items:center; justify-content:center;">
                <div class="scenario-modal__backdrop" style="position:absolute; inset:0; background:rgba(0,0,0,.4);"></div>
                <div class="scenario-modal__dialog card" style="position:relative; width:420px; max-width:90%; padding:16px; background:#fff; border-radius:8px;">
                    <div class="scenario-modal__header d-flex justify-content-between align-items-center mb-3">
                        <h5 id="scenarioModalTitle" class="m-0">Scenario</h5>
                        <button type="button" id="closeScenarioModal" class="btn-close"></button>
                    </div>
                    <form id="scenarioForm">
                        <input type="hidden" name="id" />
                        <div class="mb-3">
                            <label class="form-label">name</label>
                            <input type="text" name="name" class="form-control" required />
                        </div>
                        <div class="mb-3">
                            <label class="form-label">description</label>
                            <textarea name="description" class="form-control" rows="3"></textarea>
                        </div>
                    </form>
                    <div class="scenario-modal__footer d-flex justify-content-end gap-2">
                    <button type="button" id="submitScenarioBtn" class="btn btn-primary">save</button>
                        <button type="button" id="cancelScenarioBtn" class="btn btn-secondary">cnacel</button>
                    </div>
                </div>
            </div>
        `);
    }

    _cacheElements() {
        this._modal = $("#scenarioModal");
        this._form = $("#scenarioForm");
        this._title = $("#scenarioModalTitle");
        this._nameInput = this._form.find("input[name='name']");
        this._descInput = this._form.find("textarea[name='description']");
        this._idInput = this._form.find("input[name='id']");
        this._closeBtn = $("#closeScenarioModal");
        this._cancelBtn = $("#cancelScenarioBtn");
        this._submitBtn = $("#submitScenarioBtn");
    }

    _wireEvents() {
        this._closeBtn.on("click", () => this.close());
        this._cancelBtn.on("click", () => this.close());
        this._modal
            .find(".scenario-modal__backdrop")
            .on("click", () => this.close());

        this._submitBtn.on("click", () => {
            if (!this._form[0].checkValidity()) {
                this._form[0].reportValidity();
                return;
            }
            if (this.onSubmit) {
                this.onSubmit(this.getFormData(), this._mode);
            }
        });
    }

    // باز کردن در حالتِ «ایجاد» با فرم خالی.
    openForCreate() {
        this._mode = "create";
        this._editingId = null;
        this._title.text("Create new scenario");
        this._idInput.val("");
        this._nameInput.val("");
        this._descInput.val("");
        this._show();
    }

    // باز کردن در حالتِ «ویرایش» و پرکردن فرم از روی سناریوی انتخاب‌شده.
    openForEdit(scenario) {
        this._mode = "edit";
        this._editingId = scenario ? scenario.id : null;
        this._title.text("Scenario editing");
        this._idInput.val(scenario ? scenario.id : "");
        this._nameInput.val(scenario ? scenario.name : "");
        this._descInput.val(scenario ? scenario.description || "" : "");
        this._show();
    }

    _show() {
        this._modal.css("display", "flex");
    }

    close() {
        this._modal.css("display", "none");
    }

    // استخراج داده‌ی فرم. در حالتِ ایجاد id خالی است.
    getFormData() {
        const data = {
            name: this._nameInput.val(),
            description: this._descInput.val(),
        };
        if (this._mode === "edit" && this._editingId != null) {
            data.id = this._editingId;
        }
        return data;
    }
}
