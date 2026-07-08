// ui/HscParameterModal.js
// -----------------------------------------------------------------------------
// View مربوط به مودال «پارامترهای ثابت مسئله (HSC)».
// این کلاس مودال را نمی‌سازد؛ به مارک‌آپِ کامپوننتِ Blade (x-hsc-parameters-modal)
// که در DOM موجود است bind می‌شود — دقیقاً مثل الگوی NodePropertyPanel / AddPointModal.
//
// نگاشت اصول SOLID:
//  - SRP: فقط DOMِ همین مودال را می‌شناسد (نمایش، حالت ویرایش، استخراج داده).
//  - DIP/Observer: عملیات ذخیره (API) اینجا نیست؛ فقط رویداد onSave را می‌فرستد
//         و Controller تصمیم می‌گیرد چه اتفاقی بیفتد.
//  - LSP: اینترفیس callback-محورِ یکنواخت با بقیه‌ی Viewها.
//
// نام فیلدها: تخت (مثل B, t1, ...) و برای هزینه‌ها به‌صورت cost[key]
// تا هنگام ارسال به یک آبجکتِ تودرتوی cost تبدیل شوند.
// -----------------------------------------------------------------------------
export default class HscParameterModal {
    constructor() {
        this.onSave = null; // (data) => void
    }

    init() {
        this._cacheElements();
        this._wireEvents();
    }

    _cacheElements() {
        this._modal = $("#hscParamsModal");
        this._form = $("#hscParamsForm");
        this._closeBtn = $("#closeHscModal");
        this._editBtn = $("#hscEditBtn");
        this._saveBtn = $("#hscSaveBtn");
        this._cancelBtn = $("#hscCancelBtn");
    }

    _wireEvents() {
        this._closeBtn.on("click", () => this.close());
        this._editBtn.on("click", () => this.enterEditMode());
        this._cancelBtn.on("click", () => this.exitEditMode());
        this._modal
            .find(".hsc-modal__backdrop")
            .on("click", () => this.close());

        this._saveBtn.on("click", () => {
            if (!this._form[0].checkValidity()) {
                this._form[0].reportValidity();
                return;
            }
            if (this.onSave) this.onSave(this.getFormData());
        });
    }

    open() {
        this._modal.css("display", "flex");
    }

    close() {
        this.exitEditMode();
        this._modal.css("display", "none");
    }

    // پرکردن فیلدها از روی داده‌ی سناریو و قفل‌کردن ورودی‌ها (حالت نمایش/فقط‌خواندنی).
    display(params) {
        const flat = this._flatten(params || {});
        Object.entries(flat).forEach(([name, val]) => {
            const el = this._form.find(`[name="${name}"]`);
            if (el.length) {
                el.val(val);
                el[0].defaultValue = val; // برای بازگردانی هنگام «انصراف»
            }
        });
        this.exitEditMode();
    }

    // --- حالت‌های ویرایش ------------------------------------------------------
    enterEditMode() {
        this._fields().prop("disabled", false);
        this._editBtn.addClass("d-none");
        this._saveBtn.removeClass("d-none");
        this._cancelBtn.removeClass("d-none");
    }

    exitEditMode() {
        this._fields()
            .each(function () {
                this.value = this.defaultValue; // بازگردانی به مقدار بارگذاری‌شده
            })
            .prop("disabled", true);
        this._saveBtn.addClass("d-none");
        this._cancelBtn.addClass("d-none");
        this._editBtn.removeClass("d-none");
    }

    // --- استخراج داده --------------------------------------------------------
    // نام‌های cost[key] را به آبجکت تودرتوی cost تبدیل و اعداد را Number می‌کند.
    getFormData() {
        const data = {};
        this._fields().each(function () {
            const name = this.name;
            if (!name) return;

            const raw = this.value;
            const num = raw === "" ? null : Number(raw);
            const value = raw !== "" && !Number.isNaN(num) ? num : raw;

            data[name] = value;
        });
        return data;
    }

    // --- کمکی‌ها -------------------------------------------------------------
    _fields() {
        return this._form.find("input, textarea, select");
    }

    // تبدیل ساختارِ تودرتو (cost) به نام‌های تختِ منطبق با name فیلدها.
    _flatten(params) {
        const out = {};
        Object.entries(params).forEach(([key, val]) => {
            if (key === "cost" && val && typeof val === "object") {
                Object.entries(val).forEach(([ck, cv]) => {
                    out[`cost[${ck}]`] = cv;
                });
            } else {
                out[key] = val;
            }
        });
        return out;
    }
}
