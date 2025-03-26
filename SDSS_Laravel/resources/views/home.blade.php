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
        .dropdown-toggle{
            max-width: 20%;
        }
    </style>
</head>

<body>
    <div class="row container-fluid pe-0 mt-2 border border-2">
        {{-- Model Inputs --}}
        <div class="col-4 p-3">
            <div class="d-flex justify-content-center" style="font-weight: bold">Model Inputs</div>
            <div class="row bg-black mt-2">
                <div class="col-4 d-flex justify-content-center border border-black text-light">Affected Area</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Parameters</div>
                <div class="col-4 d-flex justify-content-center border border-black text-light">Tasks</div>
            </div>
            <div class="row mt-3">
                <div class="col-4 m-1 border border-2 border-black">Region</div>
                <div class="col-4 m-1 border border-2 border-black">APR</div>
            </div>
            <div class="row">
                <div class="col-4 btn-group m-1 d-flex justify-content-center">
                    <button class="btn btn-secondary btn-sm" type="button">
                    </button>
                    <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
                    </button>
                    <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                        <li><a class="dropdown-item" href="#">Item 1</a></li>
                        <li><a class="dropdown-item" href="#">Item 2</a></li>
                        <li><a class="dropdown-item" href="#">Item 3</a></li>
                        <li><a class="dropdown-item" href="#">Item 4</a></li>
                        <li><a class="dropdown-item" href="#">Item 5</a></li>
                        <li><a class="dropdown-item" href="#">Item 6</a></li>
                        <li><a class="dropdown-item" href="#">Item 7</a></li>
                        <li><a class="dropdown-item" href="#">Item 8</a></li>
                    </ul>
                </div>
                <div class="col-2 m-1 p-0 justify-content-center align-content-center">
                    <input class="col-12" type="text">
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-4 m-1 border border-2 border-black">City</div>
                <div class="col-4 m-1 border border-2 border-black">PP</div>
            </div>
            <div class="row">
                <div class="col-4 btn-group m-1 d-flex justify-content-center">
                    <button class="btn btn-secondary btn-sm" type="button">
                      </button>
                      <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
                      </button>
                    <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                        <li><a class="dropdown-item" href="#">Item 1</a></li>
                        <li><a class="dropdown-item" href="#">Item 2</a></li>
                        <li><a class="dropdown-item" href="#">Item 3</a></li>
                        <li><a class="dropdown-item" href="#">Item 4</a></li>
                        <li><a class="dropdown-item" href="#">Item 5</a></li>
                        <li><a class="dropdown-item" href="#">Item 6</a></li>
                        <li><a class="dropdown-item" href="#">Item 7</a></li>
                        <li><a class="dropdown-item" href="#">Item 8</a></li>
                    </ul>
                </div>
                <div class="btn-group col-4 m-1 d-flex justify-content-center">
                    <button class="btn btn-secondary btn-sm" type="button">
                    </button>
                    <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
                    </button>
                    <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                        <li><a class="dropdown-item" href="#">Item 1</a></li>
                        <li><a class="dropdown-item" href="#">Item 2</a></li>
                        <li><a class="dropdown-item" href="#">Item 3</a></li>
                        <li><a class="dropdown-item" href="#">Item 4</a></li>
                        <li><a class="dropdown-item" href="#">Item 5</a></li>
                        <li><a class="dropdown-item" href="#">Item 6</a></li>
                        <li><a class="dropdown-item" href="#">Item 7</a></li>
                        <li><a class="dropdown-item" href="#">Item 8</a></li>
                    </ul>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-4 m-1 border border-2 border-black">District</div>
                <div class="col-4 m-1 border border-2 border-black">Configuration</div>
            </div>
            <div class="row">
                <div class="col-4 btn-group m-1 d-flex justify-content-center">
                    <button class="btn btn-secondary btn-sm" type="button">
                    </button>
                    <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
                    </button>
                    <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                        <li><a class="dropdown-item" href="#">Item 1</a></li>
                        <li><a class="dropdown-item" href="#">Item 2</a></li>
                        <li><a class="dropdown-item" href="#">Item 3</a></li>
                        <li><a class="dropdown-item" href="#">Item 4</a></li>
                        <li><a class="dropdown-item" href="#">Item 5</a></li>
                        <li><a class="dropdown-item" href="#">Item 6</a></li>
                        <li><a class="dropdown-item" href="#">Item 7</a></li>
                        <li><a class="dropdown-item" href="#">Item 8</a></li>
                    </ul>
                </div>
                <div class="btn-group col-4 m-1 d-flex justify-content-center">
                    <button class="btn btn-secondary btn-sm" type="button">
                    </button>
                    <button type="button" class="btn btn-sm btn-secondary dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
                    </button>
                    <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                        <li><a class="dropdown-item" href="#">Item 1</a></li>
                        <li><a class="dropdown-item" href="#">Item 2</a></li>
                        <li><a class="dropdown-item" href="#">Item 3</a></li>
                        <li><a class="dropdown-item" href="#">Item 4</a></li>
                        <li><a class="dropdown-item" href="#">Item 5</a></li>
                        <li><a class="dropdown-item" href="#">Item 6</a></li>
                        <li><a class="dropdown-item" href="#">Item 7</a></li>
                        <li><a class="dropdown-item" href="#">Item 8</a></li>
                    </ul>
                </div>
            </div>
            <div class="row">
                <button class="col-12 mt-3" id="myButton">Run</button>
            </div>
        </div>
        {{-- Model Inputs --}}
        <div class="col-8" id="map"></div>
    </div>
</body>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="{{ asset('js/home.js') }}"></script>

</html>
