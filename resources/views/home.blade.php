@extends('app')
@push('styles')
    <style>

    </style>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    @vite('resources/css/home/home.css')
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
    {{-- map --}}
    <div class="container-fluid h-100" id="map" style="z-index: 1"></div>

    {{-- Items & Tables --}}
    <div class="position-absolute bottom-0 start-0 d-flex flex-row align-items-end w-auto mb-3" style="z-index: 2">
        {{-- Items --}}
        <div class="itemside bg-white border-1 border-stone-200 fade-out-left d-flex flex-column w-auto p-3 gap-3 justify-content-center rounded-5 ml-3 shadow"
            style="z-index: 2">
            @foreach ($node_types as $key => $value)
                <div class="items">
                    <a href="#"><img src="{{ $value['item_img'] }}" data-type="{{ $key }}"
                            class="mx-auto d-block w-12 h-12 border-2 border-stone-600 p-1 imgitems"
                            style="border-radius: 50%;" alt="..."></a>
                    <div class="textitems lh-1 d-flex justify-content-center mt-1">
                        <p class="mb-0 font-bold" style="font-size: 12px">{{ $value['elias'] }}</p>
                    </div>
                </div>
            @endforeach
        </div>
        {{-- Tables --}}
        <div class="itemstable pt-0 shadow-lg border-1 border-stone-200 d-flex flex-column d-none fade-in-right rounded-e-xl bg-white"
            style="z-index: 1; max-width: 450px;">
            <div class="border d-flex sticky-top my-2 bg-white" style="z-index: 11;">search</div>
            <div style="overflow: auto;">
                <table class="table table-hover">
                    <thead>

                    </thead>
                    <tbody>

                    </tbody>
                </table>
            </div>
        </div>
    </div>

    {{-- Node properties container --}}
        <div class="d-flex flex-column gap-3 position-absolute top-0 end-0 mt-14 mr-3">
            {{-- Node properties container --}}
            <form
                class="node-card shadow-xl p-3 rounded-5 bg-white border-1 border-stone-200"
                style="z-index: 2; width: 250px;">
                @csrf
                <!-- Header -->
                <div class="node-card-header">
                    <h6 class="font-bold">Node Properties</h6>
                </div>
                <!-- Properties -->
                <div class="node-props flex-column m-2">
                    <x-node_properties.general-properties id="generic_prop">
                        <div class="props">
                            <x-node_properties.ec-properties class="d-none {{ $node_types['ec']['class'] }}" />
                            <x-node_properties.da-properties class="d-none {{ $node_types['da']['class'] }}" />
                            <x-node_properties.tmc-properties class="d-none {{ $node_types['tmc']['class'] }}" />
                            <x-node_properties.h-properties class="d-none {{ $node_types['h']['class'] }}" />
                        </div>
                    </x-node_properties.general-properties>
                </div>
                <!-- MiniMap -->
                <div id="minimap" style="height: 100px"></div>
                <!-- Actions -->
                <div class="actions">
                    <div id="edit-actions" class="card-actions mt-2 row d-flex justify-center gap-1">
                        <button id="edit-btn"
                            class="btn btn-primary col-4 h-25 d-flex flex-row align-items-center w-auto p-1 disabled">
                            <p class="m-0 font-bold" style="font-size: 11px">Edit</p>
                            <i class="fa-solid fa-pen-to-square w-3"></i>
                        </button>
                        <button id="delete-btn"
                            class="btn btn-danger col-5 h-25 d-flex flex-row align-items-center w-auto p-1 disabled">
                            <p class="m-0 font-bold" style="font-size: 11px">Delete</p>
                            <i class="fa-solid fa-trash" style="width: 10px"></i>
                        </button>
                    </div>
                    <div id="save-actions" class="card-actions mt-2 row d-flex justify-center gap-1 d-none">
                        <button id="save-btn" class="btn btn-success col-4 h-25 d-flex flex-row align-items-center w-auto p-1">
                            <p class="m-0 font-bold" style="font-size: 11px">Save</p>
                            <i class="fa-solid fa-floppy-disk" style="width: 11px"></i>
                        </button>
                        <button id="cancel-btn" class="btn btn-danger col-5 h-25 d-flex flex-row align-items-center w-auto p-1">
                            <p class="m-0 font-bold" style="font-size: 11px">Cancel</p>
                            <i class="fa-solid fa-ban" style="width: 13px"></i>
                        </button>
                    </div>
                </div>
            </form>
            {{-- Add Point Button --}}
            <div class="add-point-btn d-flex w-auto gap-2 justify-content-center"
                style="z-index: 2; cursor: pointer;">
                <button id="addPointBtn" class="btn w-50 btn-primary d-flex flex-row align-items-center justify-content-center p-2 rounded-5 shadow-xl">
                    <i class="fa-solid fa-plus w-4 h-4"></i>
                    <p class="m-0 font-bold" style="font-size: 12px">Add Node</p>
                </button>
            </div>
        </div>

    {{-- Add Point Modal --}}
    <div id="addPointModal" class="modal fixed inset-0 flex items-center justify-center bg-black bg-opacity-50"
        style="z-index: 3;">
        <div class="bg-white rounded-5 shadow-lg w-96 p-4">
            <div class="flex justify-between items-center mb-3">
                <h5 class="text-lg font-bold">Add New Point</h5>
                <button type="button" id="closeAddModal" class="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
            </div>
            <form id="addPointForm" style="width: 100%;">
                @csrf
                <div class="row border-bottom py-1">
                    <div class="col-3 p-0">
                        <p class="mb-0 font-bold">Type</p>
                    </div>
                    <div class="col p-0">
                        <select id="pointTypeSelect" class="form-select w-full" name="type">
                            <option value="" disabled selected>Select type</option>
                            <option value="dc">Distribution Center</option>
                            <option value="ec">Shelter</option>
                            <option value="da">Affected Area</option>
                            <option value="h">Hospital</option>
                            <option value="tmc">Temporary Medical Center</option>
                        </select>
                    </div>
                </div>
                <x-node_properties.general-properties id="generic_prop_add">
                    <div class="props">
                        <x-node_properties.ec-properties class="d-none {{ $node_types['ec']['class'] }}" />
                        <x-node_properties.da-properties class="d-none {{ $node_types['da']['class'] }}" />
                        <x-node_properties.tmc-properties class="d-none {{ $node_types['tmc']['class'] }}" />
                        <x-node_properties.h-properties class="d-none {{ $node_types['h']['class'] }}" />
                    </div>
                </x-node_properties.general-properties>
                <div class="mt-3 flex justify-end gap-2">
                    <button type="button" id="saveAddBtn" class="btn btn-success">Save</button>
                    <button type="button" id="cancelAddBtn" class="btn btn-danger">Cancel</button>
                </div>
            </form>
        </div>
    </div>
@endsection

@push('scripts')
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    @vite('resources/js/home/home.js')
@endpush
