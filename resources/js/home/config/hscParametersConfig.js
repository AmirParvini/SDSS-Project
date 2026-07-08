// config/hscParametersConfig.js
// -----------------------------------------------------------------------------
// تنها منبعِ حقیقت برای «مقادیر پیش‌فرضِ پارامترهای ثابت مسئله (HSC)».
// این مقادیر هنگام ساختِ سناریوی جدید به بک‌اند ارسال می‌شوند تا در جدول
// hsc_parameters برای آن سناریو ذخیره شوند.
//
// اصل OCP: افزودن یک پارامتر جدید = فقط یک ورودی همین‌جا (+ یک فیلد در کامپوننت Blade).
// اصل SRP: این فایل هیچ منطقی ندارد؛ صرفاً پیکربندی است.
// -----------------------------------------------------------------------------
export const HSC_PARAMETER_DEFAULTS = {
    budget: 1500000, // Budget
    t1: 0.015, // Percentage distribution of severe injuries
    t2: 0.067, // Percentage distribution of mild injuries
    pua: 0.7, // Percentage of shelter area used
    rta: 17.5, // Area of relief tent (Square meter)
    rtc: 5, // Relief tent capacity (person)
    gv_speed: 20, // km/h
    av_speed: 40, // km/h
    phi_min_s: 0,
    phi_max_s: 0.9,
    ks_s: 0.1,
    tm_s: 10,
    phi_min_m: 0,
    phi_max_m: 0.9,
    ks_m: 0.1,
    tm_m: 20,
    itst: 50,
    wt: 5,
    rp_cost: 108.76,
    rpt_cost: 30,
    tmc_cost: 50000,
    shelter_cost: 50000,
    gv_cost: 50,
    av_cost: 100,
    gv_severe_capacity: 2,
    gv_moderate_capacity: 4,
    av_severe_capacity: 4,
    av_moderate_capacity: 12
};
