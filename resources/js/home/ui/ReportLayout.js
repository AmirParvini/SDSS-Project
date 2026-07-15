// ui/ReportLayout.js
// -----------------------------------------------------------------------------
// Toggles the page between "dashboard" and "report" layouts. In report mode the
// dashboard docks (items sidebar + item tables on the left, node-properties +
// add-point on the right) are hidden so the result panels have room.
//
// SRP: it only shows/hides layout regions and the active nav tab. It knows
//      nothing about the data being displayed.
// -----------------------------------------------------------------------------
export default class ReportLayout {
    constructor() {
        this._docks = $("#dashboardLeftDock, #dashboardRightDock");
        this._reportsTab = $("#reportsTab");
        this._dashboardTab = $("#dashboardTab");
    }

    enterReport() {
        this._docks.addClass("d-none");
        this._dashboardTab.removeClass("active");
        this._reportsTab.addClass("active");
    }

    enterDashboard() {
        this._docks.removeClass("d-none");
        this._reportsTab.removeClass("active");
        this._dashboardTab.addClass("active");
    }
}
