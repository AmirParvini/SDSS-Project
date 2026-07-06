// ui/NodePropertyPanel.js
// -----------------------------------------------------------------------------
// پنل سمت راست: نمایش پراپرتی‌های نود، حالت‌های ویرایش/لغو، و استخراج داده‌ی فرم.
// اصل SRP: فقط DOMِ فرم/پنل را می‌شناسد. منطق ذخیره/حذف (API) اینجا نیست؛
//         این پنل فقط داده می‌دهد و رویداد می‌فرستد (onUpdate / onDelete) و
//         Controller تصمیم می‌گیرد چه اتفاقی بیفتد.
// -----------------------------------------------------------------------------
export default class NodePropertyPanel {
    constructor() {
        this.onUpdate = null; // (formData) => void
        this.onDelete = null; // (id, type) => void
    }

    // اتصال دکمه‌های ویرایش / لغو / ذخیره / حذف.
    bind() {
        $("#edit-btn").on("click", (e) => {
            e.preventDefault();
            this.enterEditMode();
        });

        $("#cancel-btn").on("click", (e) => {
            e.preventDefault();
            this.exitEditMode();
        });

        $("#save-btn").on("click", (e) => {
            e.preventDefault();
            if (this.onUpdate) {
                this.onUpdate(this.getFormData());
            }
        });

        $("#delete-btn").on("click", (e) => {
            e.preventDefault();
            if (this.onDelete) {
                this.onDelete(this.getSelectedId(), this.getSelectedType());
            }
        });
    }

    // --- وضعیت اکشن‌ها هنگام انتخاب یک نود ------------------------------------
    // فعال‌کردن دکمه‌های ویرایش/حذف و بازگرداندن پنل به حالت غیرِ ویرایش.
    activateActions() {
        if (
            $("#edit-btn").hasClass("disabled") &&
            $("#delete-btn").hasClass("disabled")
        ) {
            $("#edit-btn").removeClass("disabled");
            $("#delete-btn").removeClass("disabled");
        }
        if (!$("#save-actions").hasClass("d-none")) {
            $("#save-actions").addClass("d-none");
            $("#edit-actions").removeClass("d-none");
            this.exitEditMode();
        }
    }

    // --- حالت‌های ویرایش ------------------------------------------------------
    enterEditMode() {
        const actions = $("#edit-btn").parents(".actions");
        actions.children().addClass("d-none");
        actions.find("#save-actions").removeClass("d-none");
        this._eachField(($field) => $field.prop("disabled", false));
    }

    exitEditMode() {
        const actions = $("#cancel-btn").parents(".actions");
        actions.children().addClass("d-none");
        actions.find("#edit-actions").removeClass("d-none");
        this._eachField(($field) => {
            // بازگرداندن به مقدار اولیه و غیرفعال‌کردن فیلد.
            $field[0].value = $field[0].defaultValue;
            $field.prop("disabled", true);
        });
    }

    // --- نمایش داده روی پنل ---------------------------------------------------
    setLatLng(latlng) {
        const lat = $("#generic_prop").find(".lat");
        const lng = $("#generic_prop").find(".lng");
        lat.val(latlng.lat);
        lat[0].defaultValue = latlng.lat;
        lng.val(latlng.lng);
        lng[0].defaultValue = latlng.lng;
    }

    showType(nodeType) {
        $(".props").children().addClass("d-none");
        $(`.${nodeType}`).removeClass("d-none");
    }

    // پرکردن فیلدها از روی props (معادلِ node_prop_display قبلی).
    displayProps(nodeType, props) {
        Object.entries(props).forEach(([prop, val]) => {
            let el = $("#generic_prop").find(`.${nodeType}`).find(`.${prop}`);
            if (el.length === 0) {
                el = $("#generic_prop").find(`.${prop}`);
            }
            if (el.is("textarea") || el.is("input")) {
                el.val(val);
            } else {
                el.text(val);
            }
            if (el[0]) el[0].defaultValue = val;
        });
    }

    // --- استخراج داده / انتخاب فعلی ------------------------------------------
    getFormData() {
        const form = $(".node-card");
        const data = {};
        form.find("*")
            .not(":hidden")
            .serializeArray()
            .forEach((item) => {
                data[item.name] = item.value;
            });
        data.id = this._readId(form.find(".id").text());
        data.type = form.find(".type").text();
        return data;
    }

    getSelectedId() {
        return this._readId($("#generic_prop").find(".id").text());
    }

    getSelectedType() {
        return $("#generic_prop").find(".type").text();
    }

    // --- پاک‌سازی پس از حذف ---------------------------------------------------
    reset() {
        $("#edit-btn").addClass("disabled");
        $("#delete-btn").addClass("disabled");

        $("#generic_prop").find(".id").text("-");
        $("#generic_prop").find(".type").text("-");

        this._eachField(($field) => {
            $field[0].value = "";
            $field[0].defaultValue = "";
            $field.prop("disabled", true);
        });

        $(".props").children().addClass("d-none");
    }

    // --- کمکی‌ها --------------------------------------------------------------
    _eachField(cb) {
        $(".node-props")
            .find("input, textarea")
            .each(function () {
                cb($(this));
            });
    }

    _readId(raw) {
        return raw && raw !== "-" ? raw : null;
    }
}
