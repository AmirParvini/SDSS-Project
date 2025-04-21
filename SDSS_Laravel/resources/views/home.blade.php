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
    <div class="row container-fluid p-0 m-0 mt-2 border border-2">
        <div id="map"></div>
        {{-- Model Inputs --}}
        <div class="p-3">
            <div class="d-flex justify-content-center" style="font-weight: bold">Model Inputs</div>
            <div class="row bg-black mt-2">
                <div class="col-4 d-flex justify-content-center border border-black text-light">Affected Area</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Parameters</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Tasks</div>
            </div>
            <div class="row mt-3">
                <div class="col-4 m-1 border border-2 border-black fw-bold">Region</div>
                <div class="col-4 m-1 border border-2 border-black fw-bold">APR</div>
            </div>
            <div class="row">
                <div class="col-4 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button class="btn btn-secondary btn-sm fw-bold" type="button">
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            <li><a class="dropdown-item">Item 1</a></li>
                            <li><a class="dropdown-item">Item 2</a></li>
                            <li><a class="dropdown-item">Item 3</a></li>
                        </ul>
                    </div>
                </div>
                <div class="col-4 m-1 p-0 justify-content-center align-content-center">
                    <input class="col-6" type="number" step="0.01" id="APR" name="APR">
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-4 m-1 border border-2 border-black fw-bold">City</div>
                <div class="col-4 m-1 border border-2 border-black fw-bold">PP(Hours)</div>
            </div>
            <div class="row">
                <div class="col-4 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button class="btn btn-secondary btn-sm fw-bold" type="button">
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            <li><a class="dropdown-item">Item 1</a></li>
                            <li><a class="dropdown-item">Item 2</a></li>
                            <li><a class="dropdown-item">Item 3</a></li>
                        </ul>
                    </div>
                </div>
                <div class="col-4 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button id="PP" class="btn btn-secondary btn-sm fw-bold" type="button">
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        @php
                            $scenarios = App\Models\Scenario::all();
                        @endphp
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            @foreach($scenarios as $scenario)
                            <li><a class="dropdown-item" data-value="{{ $scenario->{'ArrivalTime(h)'} }}">{{ data_get($scenario, 'ArrivalTime(h)') }}</a></li>
                            @endforeach
                        </ul>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-4 m-1 border border-2 border-black fw-bold">District</div>
                <div class="col-4 m-1 border border-2 border-black fw-bold">Configuration</div>
            </div>
            <div class="row">
                <div class="col-4 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button class="btn btn-secondary btn-sm fw-bold" type="button">
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            <li><a class="dropdown-item">Item 1</a></li>
                            <li><a class="dropdown-item">Item 2</a></li>
                            <li><a class="dropdown-item">Item 3</a></li>
                            <li><a class="dropdown-item">Item 4</a></li>
                            <li><a class="dropdown-item">Item 5</a></li>
                            <li><a class="dropdown-item">Item 6</a></li>
                            <li><a class="dropdown-item">Item 7</a></li>
                            <li><a class="dropdown-item">Item 8</a></li>
                        </ul>
                    </div>
                </div>
                <div class="col-4 m-1 d-flex justify-content-start">
                    <div class="col-6 btn-group">
                        <button id="Config" class="btn btn-secondary btn-sm fw-bold" type="button">
                        </button>
                        <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        </button>
                        @php
                            $configs = App\Models\Configuration::all();
                        @endphp
                        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                            @foreach($configs as $config)
                            <li><a class="dropdown-item" data-value={{ $config->Name }}>{{ $config->Name}}</a></li>
                            @endforeach
                        </ul>
                    </div>
                </div>
            </div>
            <div>
                <button class="col-12 mt-3" id="run">Solve Model</button>
            </div>
        </div>
        {{-- Model Inputs --}}
    </div>

    <div class="container-fluid text-danger" id="error_message"></div>
    <div class="container-fluid mt-2 text-center bg-primary text-white">Solution Information</div>
    <div class="container-fluid">
        @php
        $report_rows = ['Solution Status', 'Iterations', 'Solution Time (sec)', 'Total Distance', 'Total of Facalities Opened', 'Total Unmet Demand Amount'];
        @endphp
        @foreach ($report_rows as $report_row)
            <div class="row">
                <div class="col-3 border border-black d-flex align-content-center">{{ $report_row." :" }}</div>
                <div class="col-9  border border-black" id="{{ $report_row }}"></div>
            </div>
        @endforeach
    </div>


</body>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="{{ asset('js/home.js') }}"></script>

</html>
