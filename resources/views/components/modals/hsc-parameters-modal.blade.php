{{-- resources/views/components/hsc-parameters-modal.blade.php --}}
{{--
    کامپوننتِ مودالِ «پارامترهای ثابت مسئله (HSC)».
    نحوه‌ی استفاده در بلید اصلی:  <x-hsc-parameters-modal />

    - با کلیک روی دکمه‌ی #hsc_parameters باز می‌شود (منطق در ui/HscParameterModal.js).
    - ورودی‌ها به‌صورت پیش‌فرض disabled هستند؛ با دکمه‌ی «ویرایش» فعال می‌شوند.
    - مقادیر توسط JS از روی سناریوی فعال پر می‌شوند؛ این کامپوننت فقط «ساختار + UI» است.
    - عددهای داخلِ placeholder صرفاً راهنما هستند؛ مقادیر پیش‌فرضِ معتبر برای ذخیره،
      در config/hscParametersConfig.js نگه‌داری می‌شوند.
--}}
@php
    // دسته‌بندیِ تمیزِ پارامترها. افزودن پارامتر = فقط یک ردیف در همین آرایه.
    $groups = [
        [
            'title' => 'Budget and Costs',
            'icon' => '💰',
            'fields' => [
                ['name' => 'budget', 'label' => 'Budget', 'ph' => '1500000'],
                ['name' => 'rp_cost', 'label' => 'Relief package cost', 'ph' => '108.76'],
                [
                    'name' => 'rpt_cost',
                    'label' => 'Ground Transit Cost (RelifePackage)',
                    'ph' => '30',
                ],
                ['name' => 'tmc_cost', 'label' => 'TMC establishing cost', 'ph' => '50000'],
                ['name' => 'shelter_cost', 'label' => 'Shelter establishing cost', 'ph' => '50000'],
                ['name' => 'gv_cost', 'label' => 'Ground Transit Cost (Injured)', 'ph' => '50'],
                ['name' => 'av_cost', 'label' => 'Air Transit Cost (Injured)', 'ph' => '100'],
            ],
        ],
        [
            'title' => 'Injuries and Treatment',
            'icon' => '🩹',
            'fields' => [
                ['name' => 't1', 'label' => 'Severe Injury Rate (t1)', 'ph' => '0.015'],
                ['name' => 't2', 'label' => 'Moderate Injury Rate (t2)', 'ph' => '0.067'],
                ['name' => 'itst', 'label' => 'Simultaneous Treatments (itst)', 'ph' => '50'],
                ['name' => 'wt', 'label' => 'زمان انتظار (wt)', 'ph' => '5'],
            ],
        ],
        [
            'title' => 'Shelter and Relief tent',
            'icon' => '⛺',
            'fields' => [
                ['name' => 'pua', 'label' => 'Park Area Used', 'ph' => '0.7'],
                ['name' => 'rta', 'label' => 'Tent Area', 'ph' => '17.5'],
                ['name' => 'rtc', 'label' => 'Tent Capacity (person)', 'ph' => '5'],
            ],
        ],
        [
            'title' => 'Medical vehicle speed',
            'icon' => '🚑',
            'fields' => [
                ['name' => 'gv_speed', 'label' => 'Ground vehicle speed (km/h)', 'ph' => '20'],
                ['name' => 'av_speed', 'label' => 'Air vehicle speed (km/h)', 'ph' => '40'],
                ['name' => 'gv_severe_capacity', 'label' => 'Ground vehicle capacity (person, severe)', 'ph' => '2'],
                ['name' => 'gv_moderate_capacity', 'label' => 'Ground vehicle capacity (person, moderate)', 'ph' => '4'],
                ['name' => 'av_severe_capacity', 'label' => 'Air vehicle capacity (person, severe) ', 'ph' => '4'],
                ['name' => 'av_moderate_capacity', 'label' => 'Air vehicle capacity (person, moderate) ', 'ph' => '12'],
            ],
        ],
        [
            'title' => 'Severe injured parameters',
            'icon' => '🔴',
            'fields' => [
                ['name' => 'phi_min_s', 'label' => 'phi_min_s', 'ph' => '0'],
                ['name' => 'phi_max_s', 'label' => 'phi_max_s', 'ph' => '0.9'],
                ['name' => 'ks_s', 'label' => 'ks_s', 'ph' => '0.1'],
                ['name' => 'tm_s', 'label' => 'tm_s', 'ph' => '10'],
            ],
        ],
        [
            'title' => 'Moderate injured parameters',
            'icon' => '🟡',
            'fields' => [
                ['name' => 'phi_min_m', 'label' => 'phi_min_m', 'ph' => '0'],
                ['name' => 'phi_max_m', 'label' => 'phi_max_m', 'ph' => '0.9'],
                ['name' => 'ks_m', 'label' => 'ks_m', 'ph' => '0.1'],
                ['name' => 'tm_m', 'label' => 'tm_m', 'ph' => '20'],
            ],
        ],
    ];
@endphp

@once
    <style>
        .hsc-modal {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 1060;
            align-items: center;
            justify-content: center;
        }

        .hsc-modal__backdrop {
            position: absolute;
            inset: 0;
            background: rgba(15, 23, 42, .55);
            backdrop-filter: blur(2px);
        }

        .hsc-modal__dialog {
            position: relative;
            width: 760px;
            max-width: 95%;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            background: #fff;
            border-radius: 14px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, .25);
            overflow: hidden;
        }

        .hsc-modal__header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            color: #fff;
            background: linear-gradient(135deg, #0ea5e9, #2563eb);
        }

        .hsc-modal__header h5 {
            margin: 0;
            font-weight: 700;
        }

        .hsc-modal__header .btn-close {
            filter: invert(1);
            opacity: .9;
        }

        /* بدنه‌ی اسکرول‌شونده */
        .hsc-modal__body {
            padding: 18px 20px;
            overflow-y: auto;
            flex: 1 1 auto;
            background: #f8fafc;
        }

        .hsc-group {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
        }

        .hsc-group__title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }

        .hsc-group__title .icon {
            font-size: 1.1rem;
        }

        .hsc-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px 16px;
        }

        @media (max-width: 560px) {
            .hsc-grid {
                grid-template-columns: 1fr;
            }

            .hsc-modal__dialog {
                width: 96%;
            }
        }

        .hsc-field label {
            display: block;
            font-size: .8rem;
            color: #475569;
            margin-bottom: 4px;
        }

        .hsc-field input {
            width: 100%;
        }

        .hsc-field input:disabled {
            background: #f1f5f9;
            color: #0f172a;
            cursor: not-allowed;
        }

        .hsc-modal__footer {
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            padding: 14px 20px;
            border-top: 1px solid #e2e8f0;
            background: #fff;
        }
    </style>
@endonce

<div id="hscParamsModal" class="hsc-modal" aria-modal="true" role="dialog">
    <div class="hsc-modal__backdrop"></div>

    <div class="hsc-modal__dialog">
        {{-- سربرگ --}}
        <div class="hsc-modal__header">
            <h5> HSC Parameters</h5>
            <button type="button" id="closeHscModal" class="btn-close" aria-label="Close"></button>
        </div>

        {{-- بدنه‌ی اسکرول‌شونده --}}
        <form id="hscParamsForm" class="hsc-modal__body">
            @foreach ($groups as $group)
                <div class="hsc-group">
                    <div class="hsc-group__title">
                        <span class="icon">{{ $group['icon'] }}</span>
                        <span>{{ $group['title'] }}</span>
                    </div>
                    <div class="hsc-grid">
                        @foreach ($group['fields'] as $field)
                            <div class="hsc-field">
                                <label for="hsc_{{ \Illuminate\Support\Str::slug($field['name'], '_') }}">
                                    {{ $field['label'] }}
                                </label>
                                <input type="number" step="any"
                                    id="hsc_{{ \Illuminate\Support\Str::slug($field['name'], '_') }}"
                                    name="{{ $field['name'] }}" class="form-control form-control-sm"
                                    value="{{ $field['ph'] }}" disabled />
                            </div>
                        @endforeach
                    </div>
                </div>
            @endforeach
        </form>

        {{-- پاورقی: ویرایش / انصراف / ذخیره --}}
        <div class="hsc-modal__footer">
            <button type="button" id="hscEditBtn"
                class="btn btn-primary d-flex flex-row
                    align-items-center rounded-3">
                <p class="m-0 font-bold">Edit</p>
                <i class="fa-solid fa-pencil" style="width: 11px"></i>
            </button>
            <button type="button" id="hscSaveBtn"
                class="btn d-flex flex-row
                    align-items-center btn-success rounded-3">
                <p class="m-0 font-bold">Save</p>
                <i class="fa-solid fa-floppy-disk" style="width: 11px"></i>
            </button>
            <button type="button" id="hscCancelBtn"
                class="btn d-flex flex-row 
                    align-items-center btn-danger rounded-3">
                <p class="m-0 font-bold">Cancel</p>
                <i class="fa-solid fa-ban" style="width: 12px"></i>
            </button>
        </div>
    </div>
</div>
