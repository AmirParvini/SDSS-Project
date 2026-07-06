@props(['node_types'])
<div class="itemside bg-white border-1 border-stone-200 fade-out-left d-flex flex-column w-auto p-3 gap-3 justify-content-center rounded-5 ml-3 shadow"
    style="z-index: 2">
    @foreach ($node_types as $key => $value)
        <div class="items">
            <a href="#"><img src="{{ $value['item_img'] }}" data-type="{{ $key }}"
                    class="mx-auto d-block w-12 h-12 border-2 border-stone-600 p-1 imgitems" style="border-radius: 50%;"
                    alt="..."></a>
            <div class="textitems lh-1 d-flex justify-content-center mt-1">
                <p class="mb-0 font-bold" style="font-size: 12px">{{ $value['elias'] }}</p>
            </div>
        </div>
    @endforeach
</div>
