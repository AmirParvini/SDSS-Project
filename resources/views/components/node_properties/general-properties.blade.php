<div {{$attributes}}>
    @php
        $input_status = 'false'
    @endphp
    <div class="row border-bottom py-1">
        <div class="col p-0">
            <p class="mb-0">id</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <p class="mb-0 id">-</p>
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col p-0">
            <p class="mb-0">type</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <p class="mb-0 type">-</p>
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col p-0">
            <p class="mb-0">name</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <input class="mb-0 name" type="text" disabled="{{$input_status}}">
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col p-0">
            <p class="mb-0">lat</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <input class="mb-0 lat" type="number" step="0.01" disabled="{{$input_status}}">
        </div>
    </div>
    <div class="row border-bottom py-1">
        <div class="col p-0">
            <p class="mb-0">lon</p>
        </div>
        <div class="col-1">:</div>
        <div class="col p-0">
            <input class="mb-0 lng" type="number" step="0.01" disabled="{{$input_status}}">
        </div>
    </div>
    {{$slot}}
</div>
