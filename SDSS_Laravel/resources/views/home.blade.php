<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ asset('css/home.css') }}">
    <title>SDSS Panel</title>
    <style>
        #map {
            height: 70vh;
        }

        .dropdown-menu {
            max-height: 200px;
            overflow-y: auto;
        }

        .dropdown-toggle {
            max-width: 20%;
        }
    </style>
</head>

<body>
    <div class="row container-fluid p-0 m-0 mt-2 border border-1 border-black">
        <div id="map"></div>
        <div class="container-fluid mt-2" style="max-height: 250px; overflow-y: auto;">
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
        </div>
        {{-- Model Inputs --}}
        <div class="p-3">
            <div class="d-flex justify-content-center" style="font-weight: bold">Model Inputs</div>
            <div class="row bg-black mt-2">
                <div class="col-4 d-flex justify-content-center border border-black text-light">Affected Area</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Parameters</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Tasks</div>
            </div>
            <div class="row d-flex justify-content-around mt-3">
                <div class="col-3 m-1 border border-2 border-black fw-bold">Region</div>
                <div class="col-3 m-1 border border-2 border-black fw-bold">APR</div>
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
                <div class="col-3 m-1 border border-2 border-black fw-bold">City</div>
                <div class="col-3 m-1 border border-2 border-black fw-bold">PP(Hours)</div>
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
                <div class="col-3 m-1 border border-2 border-black fw-bold">District</div>
                <div class="col-3 m-1 border border-2 border-black fw-bold">Configuration</div>
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
        </div>
        {{-- Model Inputs --}}
        <div class="container-fluid text-danger" id="error_message"></div>
    </div>

    <div class="container-fluid border border-1 border-black mt-5">
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
    </div>


</body>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="{{ asset('js/home.js') }}"></script>

</html>
