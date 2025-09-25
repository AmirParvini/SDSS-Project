@php
    $point_types = [
        'IDC' => 'مرکز توزیع اقلام امدادی',
        'EC' => 'پناهگاه',
        'DA' => 'منطقه آسیب دیده',
        'H' => 'بیمارستان',
        'TMC' => 'مرکز درمانی موقت',
    ];
@endphp
<div class="modal-dialog">
    <div class="modal-content">
        <div dir="ltr" class="modal-header">
            <h5 class="modal-title" id="editPointModalLabel">افزودن مکان جدید</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
            <form id="editPointForm">
                <div class="showinput" id="pointType">
                    <label for="pointType">نوع مکان</label>
                    <input type="text" class="form-control" id="" name="id" readonly>
                </div>
                <div class="showinput mb-3" id="pointNameEdit">
                    <label for="pointNameEdit" class="form-label">نام</label>
                    <input type="text" class="form-control" id="" name="name">
                </div>
                <div class="showinput mb-3" id="pointdemandEdit">
                    <label for="pointDemandEdit" class="form-label">تقاضا</label>
                    <input type="number" class="form-control" id="" name="demand" min="0" required>
                </div>
                <div class="showinput mb-3" id="pointinjuredEdit">
                    <label for="pointInjuredEdit" class="form-label">تعداد مجروحین</label>
                    <input type="number" class="form-control" id="pointInjuredEdit" name="injured" min="0" required>
                </div>
                <div class="showinput mb-3" id="h_tmc_capacityEdit">
                    <label for="pointCapacityEdit" class="form-label">ظرفیت (نفر)</label>
                    <input type="number" class="form-control" id="h_tmc_CapacityEdit" name="capacity"
                        value="600" min="0" required>
                </div>
                <div class="showinput mb-3" id="idc_capacityEdit">
                    <label for="pointCapacityEdit" class="form-label">ظرفیت (حجم)</label>
                    <input type="number" class="form-control" id="idc_CapacityEdit" name="capacity"
                        value="20000" min="0" required>
                </div>
                <div class="showinput mb-3" id="pointcostEdit">
                    <label for="pointCostEdit" class="form-label">هزینه ثابت تاسیس (دلار)</label>
                    <input type="number" class="form-control" id="pointCostEdit" name="cost" value="50000" required>
                </div>
                <div class="row">
                    <div class="showinput col-6">
                        <label for="pointLatEdit" class="form-label">lat</label>
                        <input class="form-control" id="pointLatEdit" name="lat">
                    </div>
                    <div class="showinput col-6">
                        <label for="pointLngEdit" class="form-label">lng</label>
                        <input class="form-control" id="pointLngEdit" name="lng">
                    </div>
                </div>
            </form>
        </div>
        <div class="modal-footer">
            <button type="button" class="btn btn-danger" id="deletePoint">حذف</button>
            <button type="button" class="btn btn-primary" id="savePointEdits">ذخیره تغییرات</button>
        </div>
    </div>
</div>
