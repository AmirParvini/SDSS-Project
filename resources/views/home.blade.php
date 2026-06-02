@extends('app')
@push('styles')
    <style>
        #map{
            min-height: 100vh;
        }
        .dropdown-menu {
            max-height: 200px;
            overflow-y: auto;
        }

        .dropdown-toggle {
            width: auto;
            min-width: 200px;
        }
    </style>
@endpush

@php
    $point_types = [
        'IDC' => 'مرکز توزیع اقلام امدادی',
        'EC' => 'پناهگاه',
        'DA' => 'منطقه آسیب دیده',
        'H' => 'بیمارستان',
        'TMC' => 'مرکز درمانی موقت',
    ];
@endphp

@section('content')
    <div class="position-fixed container-fluid p-0 m-0 vh-100 overflow-hidden">
        <div id="map"></div>
    </div>
    
    <!-- Modal for adding points -->
    <div dir="rtl" class="modal fade" id="addPointModal" tabindex="-1" aria-labelledby="addPointModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div dir="ltr" class="modal-header">
                    <h5 class="modal-title" id="addPointModalLabel">افزودن مکان جدید</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <form id="addPointForm">
                        <select class="form-select" aria-label="Default select example">
                            @foreach ($point_types as $key => $value)
                                <option value="{{ $key }}">{{ $value }}</option>
                            @endforeach
                        </select>
                        <div class="mb-3" id="name">
                            <label for="pointName" class="form-label">نام</label>
                            <input type="text" class="form-control" id="pointName" name="name" required>
                        </div>
                        <div class="mb-3 d-none" id="demand">
                            <label for="pointDemand" class="form-label">مساحت پناهگاه</label>
                            <input type="number" class="form-control" id="pointDemand" name="demand" min="0"
                                required>
                        </div>
                        <div class="mb-3 d-none" id="injured">
                            <label for="pointInjured" class="form-label">جمعیت تحت تاثر</label>
                            <input type="number" class="form-control" id="pointInjured" name="injured" min="0"
                                required>
                        </div>
                        <div class="mb-3 d-none" id="h_tmc_capacity">
                            <label for="pointCapacity" class="form-label">ظرفیت (نفر)</label>
                            <input type="number" class="form-control read-only" id="h_tmc_Capacity" name="capacity"
                                value="600" min="0" required>
                        </div>
                        <div class="mb-3" id="idc_capacity">
                            <label for="pointCapacity" class="form-label">ظرفیت (حجم)</label>
                            <input type="number" class="form-control read-only" id="idc_Capacity" name="capacity"
                                value="20000" min="0" required>
                        </div>
                        <div class="mb-3" id="cost">
                            <label for="pointCost" class="form-label">هزینه ثابت تاسیس (دلار)</label>
                            <input type="number" class="form-control" id="pointCost" name="cost" value="50000" required>
                        </div>
                        <div class="row">
                            <div class="col-6">
                                <label for="lat" class="form-label">lat</label>
                                <input class="form-control" id="pointLat" name="Lat">
                            </div>
                            <div class="col-6">
                                <label for="lng" class="form-label">lng</label>
                                <input class="form-control" id="pointLng" name="Lng">
                            </div>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">لغو</button>
                    <button type="button" class="btn btn-primary" id="savePoint">ذخیره</button>
                </div>
            </div>
        </div>
    </div>
    <div dir="rtl" id="editPointModal" class="modal fade" tabindex="-1" aria-labelledby="editPointModalLabel" aria-hidden="true">
        <x-edit_point></x-edit_point>
    </div>
@endsection

@push('scripts')
    <script>
        Points = {
            IDC : @json($idc_points),
            EC : @json($ec_points),
            DA : @json($da_points),
            TMC : @json($tmc_points),
            H : @json($H_points)
        };
    </script>
    @vite('resources/js/home/home.js')
@endpush


