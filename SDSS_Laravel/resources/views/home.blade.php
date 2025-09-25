@extends('app')
@push('styles')
    <style>
        #map {
            height: 70vh;
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
    <div class="row container-fluid p-0 m-0 mt-2 border border-1 border-black">
        <div id="map"></div>
        {{-- <div class="container-fluid mt-2" style="max-height: 250px; overflow-y: auto;">
            <table id="commo_shipped_table" class="table table-hover mt-2 d-none">
                <thead class="sticky-header">
                    <tr>
                        <th scope="col">From</th>
                        <th scope="col">To</th>
                        <th scope="col">Quantity of Water shipped</th>
                        <th scope="col">Quantity of Food shipped</th>
                        <th scope="col">Quantity of Medical_Kits shipped</th>
                        <th scope="col">Quantity of Shelter shipped</th>
                        <th scope="col">Total Commodities shipped</th>
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
        </div> --}}
        {{-- Model Inputs --}}
        {{-- <div class="p-3">
            <div class="d-flex justify-content-center" style="font-weight: bold">Model Inputs</div>
            <div class="row bg-black mt-2">
                <div class="col-4 d-flex justify-content-center border border-black text-light">Affected Area</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Parameters</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Tasks</div>
            </div>
            <div class="row d-flex justify-content-around mt-3">
                <div class="col-3 m-1 border-2 border-black fw-bold">Region</div>
                <div class="col-3 m-1 border-2 border-black fw-bold">APR</div>
                <div class="col-3 m-1 d-flex justify-content-center">
                    <button class="w-50" id="update_data">Update Data</button>
                </div>
            </div>
            <div class="row d-flex justify-content-around">
                <div class="col-3 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button id="Region" class="btn btn-secondary btn-sm fw-bold" type="button">
                            تهران
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            <li><a class="dropdown-item">تهران</a></li>
                        </ul>
                    </div>
                </div>
                <div class="col-3 m-1 p-0 justify-content-center align-content-center">
                    <input class="col-6 ms-3" type="number" step="0.01" id="APR" name="APR" min="0"
                        value="0.95">
                </div>
                <div class="col-3 m-1"></div>
            </div>
            <div class="row mt-3 d-flex justify-content-around">
                <div class="col-3 m-1 border-2 border-black fw-bold">City</div>
                <div class="col-3 m-1 border-2 border-black fw-bold">PP(Hours)</div>
                <div class="col-3 m-1 d-flex justify-content-center">
                    <button class="w-50" id="create_inputs">Create Inputs</button>
                </div>
            </div>
            <div class="row d-flex justify-content-around">
                <div class="col-3 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button class="btn btn-secondary btn-sm fw-bold" type="button">
                            تهران
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            <li><a class="dropdown-item">تهران</a></li>
                        </ul>
                    </div>
                </div>
                <div class="col-3 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button id="PP" class="btn btn-secondary btn-sm fw-bold" type="button">
                            8
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        @php
                            $scenarios = App\Models\Scenario::all();
                        @endphp
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            @foreach ($scenarios as $scenario)
                                <li><a class="dropdown-item"
                                        data-value="{{ $scenario->{'ArrivalTime(h)'} }}">{{ data_get($scenario, 'ArrivalTime(h)') }}</a>
                                </li>
                            @endforeach
                        </ul>
                    </div>
                </div>
                <div class="col-3"></div>
            </div>
            <div class="row d-flex justify-content-around mt-3">
                <div class="col-3 m-1 border-2 border-black fw-bold">District</div>
                <div class="col-3 m-1 border-2 border-black fw-bold">Configuration</div>
                <div class="col-3 m-1 d-flex justify-content-center">
                    <button class="w-50" id="run">Solve Model</button>
                </div>
            </div>
            <div class="row d-flex justify-content-around">
                <div class="col-3 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button id="district" class="btn btn-secondary btn-sm fw-bold" type="button">
                            4
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        @php
                            $districts = range(1, 22);
                        @endphp
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            @foreach ($districts as $district)
                                <li><a class="dropdown-item">{{ $district }}</a></li>
                            @endforeach
                        </ul>
                    </div>
                </div>
                <div class="col-3 m-1">
                    <div class="col-6 btn-group">
                        @php
                            $configs = App\Models\Configuration::all();
                        @endphp
                        <button id="Config" class="btn btn-secondary btn-sm fw-bold" type="button">
                            {{ $configs[0]->Name }}
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            @foreach ($configs as $config)
                                <li><a class="dropdown-item" data-value={{ $config->Name }}>{{ $config->Name }}</a>
                                </li>
                            @endforeach
                        </ul>
                    </div>
                </div>
                <div class="col-3"></div>
            </div>
        </div> --}}
        {{-- Model Inputs --}}
        <div class="container-fluid text-danger" id="error_message"></div>
    </div>

    {{-- <div class="container-fluid border border-1 border-black mt-5">
        <div class="d-flex justify-content-center mt-2 mb-2" style="font-weight: bold">Model Reports</div>
        <div class="container-fluid">
            <!-- Nav tabs -->
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="solution-tab" data-bs-toggle="tab"
                        data-bs-target="#solution_info_container" type="button" role="tab">Solution
                        Information</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="profile-tab" data-bs-toggle="tab"
                        data-bs-target="#inventory_decision_container" type="button" role="tab">Inventory
                        Decisions</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="contact-tab" data-bs-toggle="tab" data-bs-target="#service_decision_container"
                        type="button" role="tab">Service Decisions</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="contact-tab" data-bs-toggle="tab" data-bs-target="#Additional_inventory_container"
                        type="button" role="tab">Additional inventory</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="contact-tab" data-bs-toggle="tab" data-bs-target="#unmet_demand_container"
                        type="button" role="tab">Unmet demand</button>
                </li>
            </ul>

            <!-- Tab content -->
            @php
                $solution_report_rows = [
                    'Solution Status',
                    'Iterations',
                    'Solution Time (sec)',
                    'Total Distance',
                    'Total LDC Used',
                    'Total Unmet Demand Amount',
                    'Total Additional Inventory',
                ];
                $inventory_decision_rows = [
                    'Inventory Decision',
                    'Inventory Amount',
                    'Inventory Cost',
                    'Total Inventory Cost',
                ];
            @endphp
            <div class="tab-content mt-3" id="myTabContent">
                <div class="tab-pane fade show active" id="solution_info_container" role="tabpanel">
                    <div class="container-fluid mt-2 text-center bg-primary text-white border border-black mb-1">
                        Solution
                        Information</div>
                    <div class="report_container container-fluid">
                        @foreach ($solution_report_rows as $solution_report_row)
                            <div class="row">
                                <div class="col-3 border border-black d-flex align-content-center mb-1 fw-bold">
                                    {{ $solution_report_row . ' :' }}</div>
                                <div class="col-9  border border-black mb-1" id="{{ $solution_report_row }}"></div>
                            </div>
                        @endforeach
                    </div>
                </div>
                <div class="tab-pane fade" id="inventory_decision_container" role="tabpanel">
                    <div class="container-fluid mt-2 text-center bg-primary text-white border border-black mb-1">
                        Inventory Decisions</div>
                    <div class="report_container container-fluid">
                        <table id="inventory_table" class="table table-hover">
                            <thead class="sticky-header">
                                <tr><input class="me-1" type="text" id="CMDsearchInput"
                                        placeholder="CMD Search..." onkeyup="filterInventoryDecisionTable(event)" />
                                    <input type="text" id="LDCsearchInput" placeholder="LDC Search..."
                                        onkeyup="filterInventoryDecisionTable(event)" />
                                </tr>
                                <tr>
                                    <th scope="col">CMD Name</th>
                                    <th scope="col">LDC Name</th>
                                    <th scope="col">Quantity of Water Storage</th>
                                    <th scope="col">Quantity of Food Storage</th>
                                    <th scope="col">Quantity of Medical_Kits Storage</th>
                                    <th scope="col">Quantity of Shelter Storage</th>
                                    <th scope="col">Total Commodities Storage</th>
                                </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="tab-pane fade" id="service_decision_container" role="tabpanel">
                    <div class="container-fluid mt-2 text-center bg-primary text-white border border-black mb-1">
                        Service Decisions</div>
                    <div class="report_container container-fluid">
                        <table id="service_table" class="table table-hover">
                            <thead class="sticky-header">
                                <tr><input class="me-1" type="text" id="LDCsearchInput"
                                        placeholder="LDC Search..." onkeyup="filterServiceDecisionTable(event)" />
                                    <input type="text" id="ECsearchInput" placeholder="EC Search..."
                                        onkeyup="filterServiceDecisionTable(event)" />
                                </tr>
                                <tr>
                                    <th scope="col">LDC Name</th>
                                    <th scope="col">EC Name</th>
                                    <th scope="col">Quantity of Water shipped</th>
                                    <th scope="col">Quantity of Food shipped</th>
                                    <th scope="col">Quantity of Medical_Kits shipped</th>
                                    <th scope="col">Quantity of Shelter shipped</th>
                                    <th scope="col">Total Commodities shipped</th>
                                </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="tab-pane fade" id="Additional_inventory_container" role="tabpanel">
                    <div class="container-fluid mt-2 text-center bg-primary text-white border border-black mb-1">
                        Additional inventory Info</div>
                    <div class="report_container container-fluid">
                        <table id="additional_inventory_table" class="table table-striped">
                            <thead class="sticky-header">
                                <tr>
                                    <th scope="col">LDC Name</th>
                                    <th scope="col">Amount of excess Water</th>
                                    <th scope="col">Amount of excess Food</th>
                                    <th scope="col">Amount of excess Medical_kit</th>
                                    <th scope="col">Amount of excess Shelter</th>
                                    <th scope="col">Total inventory amount</th>
                                </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="tab-pane fade" id="unmet_demand_container" role="tabpanel">
                    <div class="container-fluid mt-2 text-center bg-primary text-white border border-black mb-1">
                        Unmet demand Info</div>
                    <div class="report_container container-fluid">
                        <table id="unmet_demand_table" class="table table-striped">
                            <thead class="sticky-header">
                                <tr>
                                    <th scope="col">EC Name</th>
                                    <th scope="col">Unmet Water demand</th>
                                    <th scope="col">Unmet Food demand</th>
                                    <th scope="col">Unmet Medical_kit demand</th>
                                    <th scope="col">Unmet Shelter demand</th>
                                    <th scope="col">Total Unmet demand</th>
                                </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </div>
    </div> --}}


    <!-- Modal for adding damaged area -->
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
                            <label for="pointDemand" class="form-label">تقاضا</label>
                            <input type="number" class="form-control" id="pointDemand" name="demand" min="0"
                                required>
                        </div>
                        <div class="mb-3 d-none" id="injured">
                            <label for="pointInjured" class="form-label">تعداد مجروحین</label>
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


