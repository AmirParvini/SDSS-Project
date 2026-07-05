// config/nodeConfig.js
// -----------------------------------------------------------------------------
// تنها منبع حقیقت برای «انواع نقطه» (node types).
// اصل OCP: برای افزودن یک نوع جدید فقط همین‌جا یک آیکون اضافه می‌کنی؛
// نه View، نه Controller و نه Service نیازی به تغییر ندارند.
// -----------------------------------------------------------------------------
import {
    da_icon,
    dc_icon,
    ec_icon,
    h_icon,
    puls_icon,
    tmc_icon,
} from "./mapIcons.js";

export const NODE_ICONS = {
    dc: dc_icon,
    da: da_icon,
    ec: ec_icon,
    tmc: tmc_icon,
    h: h_icon,
};

// آیکون مارکرِ ضربان‌دار (انتخاب فعلی روی نقشه)
export const PULSING_ICON = puls_icon;

// لیست انواع نقطه، مشتق‌شده از NODE_ICONS تا هرگز از هم جدا نیفتند.
export const NODE_TYPES = Object.keys(NODE_ICONS);

// آدرس پایه‌ی سرویس بک‌اند.
export const API_BASE_URL = "http://127.0.0.1:8000";
