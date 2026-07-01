@extends('app')
@push('styles')
    <style>
        .dropdown-menu {
            max-height: 200px;
            overflow-y: auto;
        }

        .dropdown-toggle {
            width: auto;
            min-width: 200px;
        }
    </style>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    @vite('resources/css/home/home.css')
@endpush
@section('content')
    {{-- map --}}
    <div class="container-fluid h-100" id="map" style="z-index: 1"></div>

    {{-- Items & Tables --}}
    <div class="position-absolute bottom-0 start-0 d-flex flex-row w-auto mb-3" style="z-index: 2">
        {{-- Items --}}
        <div class="itemside bg-white border-1 border-stone-200 fade-out-left d-flex flex-column w-auto p-3 gap-3 justify-content-center rounded-5 ml-3 shadow"
            style="z-index: 2">
            @for ($i = 0; $i < 5; $i++)
                <div class="items">
                    <a href="#"><img src="{{ asset('/images/warehouse-with-truck.svg') }}"
                            class="mx-auto d-block w-12 h-12 border-2 border-stone-600 p-1 imgitems"
                            style="border-radius: 50%;" alt="..." id="dc"></a>
                    <div class="textitems lh-1 d-flex justify-content-center mt-1">
                        <p class="mb-0 font-bold" style="font-size: 12px">Distribution center</p>
                    </div>
                </div>
            @endfor
        </div>
        {{-- Tables --}}
        <div class="itemstable shadow-lg border-1 border-stone-200 d-flex flex-column d-none fade-in-right rounded-e-xl bg-white"
            style="z-index: 1">
            <div class="border d-flex">search</div>
            <table class="table table-hover mt-2">
                <thead>
                    <tr>
                        <th scope="col">
                            <p>Id</p>
                        </th>
                        <th scope="col">
                            <p>Name</p>
                        </th>
                        <th scope="col">
                            <p>Capacity</p>
                        </th>
                        <th scope="col">
                            <p>Cost</p>
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th scope="row">
                            <p>1</p>
                        </th>
                        <td>
                            <p>Mark</p>
                        </td>
                        <td>
                            <p>Otto</p>
                        </td>
                        <td>
                            <p>@mdb</p>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    {{-- Node properties container --}}
    <div class="node-card shadow-xl position-absolute top-0 end-0 mt-14 mr-3 p-3 rounded-5 bg-white border-1 border-stone-200"
        style="z-index: 2; width: 250px;">
        <!-- Header -->
        <div class="node-card-header">
            <h6 class="font-bold">Node Properties</h6>
        </div>
        <!-- Properties -->
        <div class="node-props flex-column m-2">
            <x-node_properties.general-properties id="generic_prop">
            <div class="props">
                <x-node_properties.ec-properties class="d-none ec" />
                <x-node_properties.da-properties class="d-none da" />
                <x-node_properties.tmc-properties class="d-none tmc" />
                <x-node_properties.h-properties class="d-none h" />
            </div>
        </x-node_properties.general-properties>
        </div>
        <!-- MiniMap -->
        <div id="minimap" style="height: 100px"></div>
        <!-- Actions -->
        <div class="card-actions mt-2 row d-flex justify-center gap-1">
            <button class="btn btn-primary col-4 h-25 d-flex flex-row align-items-center w-auto p-1">
                <p class="m-0 font-bold" style="font-size: 11px">Edit</p>
                <i class="fa-solid fa-pen-to-square w-3"></i>
            </button>
            <button class="btn btn-danger col-5 h-25 d-flex flex-row align-items-center w-auto p-1">
                <p class="m-0 font-bold" style="font-size: 11px">Delete</p>
                <i class="fa-solid fa-trash" style="width: 10px"></i>
            </button>
        </div>
    </div>

    <!-- Modal for adding points -->
    {{-- <div dir="rtl" class="modal fade" id="addPointModal" tabindex="-1" aria-labelledby="addPointModalLabel" aria-hidden="true">
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
    </div> --}}
@endsection

@push('scripts')
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    @vite('resources/js/home/home.js')
@endpush
