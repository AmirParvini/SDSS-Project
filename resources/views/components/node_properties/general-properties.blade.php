<div {{ $attributes }}>
    @php
        $input_status = 'false';
    @endphp
    <div class="row border-bottom py-1">
        <div class="col-2 p-0">
            <p class="mb-0 font-bold">id</p>
        </div>
        <div class="col-2">:</div>
        <div class="col p-0">
            <p class="mb-0 id">-</p>
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col-2 p-0">
            <p class="mb-0 font-bold">type</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <p class="mb-0 type">-</p>
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col-2 p-0">
            <p class="mb-0 font-bold">name</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <textarea name="name" class="mb-0 name node_name w-100" style="direction: rtl;" disabled="{{ $input_status }}"></textarea>
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col-2 p-0">
            <p class="mb-0 font-bold">lat</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <input name="lat" class="mb-0 lat w-100" type="number" disabled="{{ $input_status }}">
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col-2 p-0">
            <p class="mb-0 font-bold">lon</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <input name="lng" class="mb-0 lng w-100" type="number" disabled="{{ $input_status }}">
        </div>
    </div>
    {{ $slot }}
</div>
