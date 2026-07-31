@extends('app')
@push('styles')
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    @vite('resources/css/home/home.css')
    @vite('resources/css/home/results.css')
@endpush

@php
    $node_types = [
        'dc' => [
            'class' => 'dc',
            'item_img' => asset('/images/warehouse-with-truck.svg'),
            'elias' => 'Distribution Center',
        ],
        'ec' => [
            'class' => 'ec',
            'item_img' => asset('/images/sturdy-house-shelter.svg'),
            'elias' => 'shelter',
        ],
        'da' => [
            'class' => 'da',
            'item_img' => asset('/images/broken-building.svg'),
            'elias' => 'Affected Area',
        ],
        'h' => [
            'class' => 'h',
            'item_img' => asset('/images/hospital-building-cross.svg'),
            'elias' => 'Hospital',
        ],
        'tmc' => [
            'class' => 'tmc',
            'item_img' => asset('/images/tent-with-medical-cross.svg'),
            'elias' => 'Temporary Medical Center',
        ],
    ];
@endphp

@section('content')
    <div class="container-fluid p-0 m-0 vh-100 d-flex flex-column overflow-hidden">
        {{-- Header --}}
        <div class="d-flex flex-row justify-content-start gap-3 px-3 py-2 w-100 rounded-bottom-4 pl-5 bg-white shadow"
            style="z-index: 2; pointer-events: auto">
            <div style="pointer-events: auto">
                <ul class="nav nav-underline">
                    <li class="nav-item pr-4">
                        <a id="dashboardTab" class="nav-link active" aria-current="page" href="#">Dashboard</a>
                    </li>
                    <li class="nav-item pr-4">
                        <a id="reportsTab" class="nav-link" href="#">Reports</a>
                    </li>
                </ul>
            </div>
            <div style="pointer-events: auto">
                <button id="runModelBtn"
                    class="btn btn-primary d-flex flex-row align-items-center h-100 justify-content-center rounded-5">
                    <span class="run-model__idle">
                        <i class="fa fa-play w-4 h-4"></i>
                        <p class="m-0 font-bold" style="font-size: 12px">Run Model</p>
                    </span>
                    <span class="run-model__busy">
                        <span class="run-spinner"></span>
                        <p class="m-0 font-bold" style="font-size: 12px">Running...</p>
                    </span>
                </button>
            </div>
            <div style="pointer-events: auto" class="d-none">
                <button id="cancelRunningBtn" class="btn btn-danger h-100 rounded-5">
                    <span class="d-flex flex-row align-items-center">
                        <i class="fa fa-times"></i>
                        <p class="m-0 font-bold" style="font-size: 12px">Cancel</p>
                    </span>
                </button>
            </div>
            <div>
                <button id="hsc_parameters"
                    class="btn btn-primary d-flex flex-row align-items-center h-100 justify-content-center rounded-4">
                    <p class="font-bold" style="font-size: 12px;">
                        HSC Parameters</p>
                </button>
            </div>
            {{-- Scenario Selector --}}
            <div id="scenarioBar">
                <div class="scenario-bar d-flex align-items-center gap-1 h-100">
                    <select id="scenarioSelect"
                        class="form-select scenario-select bg-primary h-100 focus
                    text-white fw-bold"
                        style=" font-size: 12px; pointer-events: auto;">
                        <option value="" disabled selected>Scenario Selection</option>
                    </select>

                    <button id="createScenarioBtn" type="button" class="btn btn-primary p-0 px-1 rounded-5">
                        <i class="fa fa-plus" aria-hidden="true" style="width: 16px"></i>
                    </button>
                    <button id="editScenarioBtn" type="button" class="btn btn-primary p-0 px-1 rounded-5">
                        <i class="fa-solid fa-pencil" aria-hidden="true" style="width: 15px"></i>
                    </button>
                </div>
            </div>
        </div>
        {{-- Contents --}}
        <div class="position-absolute dashboard container-fluid p-0 m-0 vh-100" style="z-index: 1">
            {{-- map --}}
            <div class="container-fluid h-100" id="map" style="z-index: 1"></div>

            {{-- Items & Tables --}}
            <div id="dashboardLeftDock"
                class="position-absolute bottom-0 start-0 d-flex flex-row align-items-end w-auto mb-3" style="z-index: 2">
                <x-items :node_types="$node_types" />
                <x-tables />
            </div>

            <div id="dashboardRightDock" class="d-flex flex-column gap-3 position-absolute bottom-0 end-0 mb-3 mr-3">
                {{-- Node properties container --}}
                <x-node_properties.main-container :node_types="$node_types" />
                {{-- Add Point Button --}}
                <div class="add-point-btn d-flex w-auto gap-2 justify-content-center"
                    style="z-index: 2; pointer-events: none">
                    <button id="addPointBtn"
                        class="btn w-50 btn-primary d-flex flex-row disabled
                align-items-center justify-content-center p-2 rounded-5 shadow-xl">
                        <i class="fa-solid fa-plus w-4 h-4"></i>
                        <p class="m-0 font-bold" style="font-size: 12px">Add Point</p>
                    </button>
                </div>
            </div>

            {{-- Result panels (visible only in report mode) --}}
            <x-results.costs-report />
            <x-results.pareto-table />
            <x-results.allocations-table />
        </div>
        <x-modals.add-point-modal style="z-index: 3;" :node_types="$node_types" />
        <x-modals.hsc-parameters-modal id="hscParamsModal" style="z-index: 3;" />
    </div>
@endsection

@push('scripts')
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    @vite('resources/js/home/home.js')
@endpush
