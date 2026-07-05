// map_init.js  (barrel سازگاری — اختیاری / موقت)
// -----------------------------------------------------------------------------
// فقط برای اینکه فایل‌های دیگرِ برنامه که هنوز از "./map_init.js" import می‌کنند
// نشکنند. توصیه: importها را به‌تدریج به مبدأ واقعی منتقل کن و این فایل را حذف کن:
//   - نقشه/مینی‌مپ/tile → ./map/mapInstance.js
//   - آیکون‌ها          → ./config/mapIcons.js
// -----------------------------------------------------------------------------
export { map, minimap, map_tiles } from "./map/mapInstance.js";
export {
    dc_icon,
    ec_icon,
    h_icon,
    da_icon,
    tmc_icon,
    puls_icon,
} from "./config/mapIcons.js";
