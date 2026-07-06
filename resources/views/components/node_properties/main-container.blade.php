@props(['node_types'])
<div class="node-card shadow-xl p-3 rounded-5 bg-white border-1 border-stone-200" style="z-index: 2; width: 250px;">
    <form>
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
                    class="btn btn-primary col-4 h-25 d-flex flex-row align-items-center
                            w-auto p-1 rounded-3 disabled">
                    <p class="m-0 font-bold" style="font-size: 11px">Edit</p>
                    <i class="fa-solid fa-pen-to-square w-3"></i>
                </button>
                <button id="delete-btn"
                    class="btn btn-danger col-5 h-25 d-flex flex-row align-items-center
                            w-auto p-1 rounded-3 disabled">
                    <p class="m-0 font-bold" style="font-size: 11px">Delete</p>
                    <i class="fa-solid fa-trash" style="width: 10px"></i>
                </button>
            </div>
            <div id="save-actions" class="card-actions mt-2 row d-flex justify-center gap-1
                    d-none">
                <button id="save-btn"
                    class="btn btn-success col-4 h-25 d-flex flex-row
                        align-items-center w-auto p-1 rounded-3">
                    <p class="m-0 font-bold" style="font-size: 11px">Save</p>
                    <i class="fa-solid fa-floppy-disk" style="width: 11px"></i>
                </button>
                <button id="cancel-btn"
                    class="btn btn-danger col-5 h-25 d-flex flex-row
                        align-items-center w-auto p-1 rounded-3">
                    <p class="m-0 font-bold" style="font-size: 11px">Cancel</p>
                    <i class="fa-solid fa-ban" style="width: 13px"></i>
                </button>
            </div>
        </div>
    </form>
</div>
