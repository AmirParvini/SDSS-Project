@props(['node_types'])
<div {{ $attributes }}>
    <div id="addPointModal" class="modal fixed inset-0 flex items-center justify-center bg-black bg-opacity-50"
        style="z-index: 3;">
        <div class="bg-white rounded-5 shadow-lg w-96 p-4">
            <div class="flex justify-between items-center mb-3">
                <h5 class="text-lg font-bold">Add New Point</h5>
                <button type="button" id="closeAddModal"
                    class="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
            </div>
            <form class="px-3" id="addPointForm" style="width: 100%;">
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
                    <button type="button" id="saveAddBtn"
                        class="btn d-flex flex-row
                    align-items-center btn-success rounded-3">
                        <i class="fa-solid fa-floppy-disk" style="width: 11px"></i>
                        <p class="m-0 font-bold">Save</p>
                    </button>
                    <button type="button" id="cancelAddBtn"
                        class="btn d-flex flex-row 
                    align-items-center btn-danger rounded-3">
                        <i class="fa-solid fa-times" style="width: 11px"></i>
                        <p class="m-0 font-bold">Cancel</p>
                    </button>
                </div>
            </form>
        </div>
    </div>
</div>
