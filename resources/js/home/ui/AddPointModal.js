// ui/AddPointModal.js
// -----------------------------------------------------------------------------
// Modal for adding new points: handles type selection and dynamic property display.
// Follows SOLID principles:
//  - SRP: Only handles DOM for the add point modal.
//  - OCP: New point types can be added via config/nodeConfig.js.
// -----------------------------------------------------------------------------
export default class AddPointModal {
    constructor() {
        this.onCreate = null; // (formData) => void
        this.onCancel = null; // () => void
        this._propContainers = {};
    }

    // Initialize modal elements and event listeners
    init() {
        this._cacheElements();
        this._wireEvents();
    }

    // Cache DOM elements
    _cacheElements() {
        this._modal = $("#addPointModal");
        this._typeSelect = $("#pointTypeSelect");
        this._closeBtn = $("#closeAddModal");
        this._cancelBtn = $("#cancelAddBtn");
        this._saveBtn = $("#saveAddBtn");

        // Cache property containers
        this._propContainers = {
            ec: $("#addPointModal").find(`.${this._getTypeClass('ec')}`),
            da: $("#addPointModal").find(`.${this._getTypeClass('da')}`),
            tmc: $("#addPointModal").find(`.${this._getTypeClass('tmc')}`),
            h: $("#addPointModal").find(`.${this._getTypeClass('h')}`)
        };
    }

    // Get CSS class for each type
    _getTypeClass(type) {
        const typeClasses = {
            dc: 'dc',
            ec: 'ec',
            da: 'da',
            h: 'h',
            tmc: 'tmc'
        };
        return typeClasses[type] || type;
    }

    // Wire event listeners
    _wireEvents() {
        // Type selection change - show relevant properties
        this._typeSelect.on("change", () => {
            this._showPropertiesForType(this._typeSelect.val());
        });

        // Close modal handlers
        this._closeBtn.on("click", () => this.hide());
        this._cancelBtn.on("click", () => this.hide());

        // Save button handler
        this._saveBtn.on("click", () => {
            if (this.onCreate) {
                this.onCreate(this.getFormData());
            }
        });
    }

    // Show properties based on selected type
    _showPropertiesForType(type) {
        this._modal.find(".type").text(type);

        // Hide all property containers
        Object.values(this._propContainers).forEach($el => {
            $el.addClass("d-none");
        });

        // Show the selected type's properties
        if (this._propContainers[type]) {
            this._propContainers[type].removeClass("d-none");
        }
    }

    // Enter add point mode - map click will open modal
    enterAddMode() {
        this._addMode = true;
    }

    exitAddMode() {
        this._addMode = false;
    }

    isInAddMode() {
        return this._addMode || false;
    }

    // Show modal with optional lat/lng
    show(latlng = null) {
        this._modal.css("display", "flex");
        
        // Reset form
        this._typeSelect.val("");
        this._showPropertiesForType(null);

        // Enable all inputs in the modal
        this._modal.find("input, textarea").prop("disabled", false);

        // Set lat/lng if provided
        if (latlng) {
            this._modal.find("input[name='lat']").val(latlng.lat);
            this._modal.find("input[name='lng']").val(latlng.lng);
        }
        
        // Exit add mode after showing modal
        this.exitAddMode();
    }

    // Hide modal
    hide() {
        this._modal.css("display", "none");
        this._resetForm();
    }

    // Reset form to initial state
    _resetForm() {
        this._typeSelect.val("");
        this._modal.find("input, textarea").val("");
        this._showPropertiesForType(null);
    }

    // Get form data
    getFormData() {
        const form = $("#addPointForm");
        const data = {};
        
        // Get all inputs, textareas, and selects that are not hidden
        form.find("input, textarea, select").each(function () {
            if (this.name && !$(this).closest(".d-none").length) {
                data[this.name] = $(this).val();
            }
        });

        return data;
    }
}
