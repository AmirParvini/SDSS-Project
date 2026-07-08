// services/HscParameterService.js
// -----------------------------------------------------------------------------
// لایه‌ی دسترسی به داده (Repository) برای «پارامترهای ثابت مسئله (HSC)».
// تنها مسئولیتش: صحبت با بک‌اند درباره‌ی hsc_parameters.
//
// نگاشت اصول SOLID:
//  - SRP: فقط خواندن/ذخیره/به‌روزرسانیِ پارامترها. هیچ منطق DOM/نقشه‌ای ندارد.
//  - DIP: axios از بیرون تزریق می‌شود (constructor injection) تا تست‌پذیر بماند.
//  - ISP: فقط سه متدِ موردنیازِ کاربرِ HSC را بیرون می‌دهد.
//
// قرارداد با بک‌اند (همه روی «سناریوی فعال» عمل می‌کنند، مثل load_data):
//   GET  /hsc_parameters  -> { hsc_parameters: {...} }
//   POST /hsc_parameters  -> { success, message }   (ذخیره‌ی مقادیر پیش‌فرض هنگام ساخت سناریو)
//   PUT  /hsc_parameters  -> { success, message }   (به‌روزرسانی پس از ویرایش)
// -----------------------------------------------------------------------------
import axios from "axios";
import { API_BASE_URL } from "../config/nodeConfig.js";

export default class HscParameterService {
    constructor(httpClient = axios) {
        this.http = httpClient;
    }

    // پارامترهای HSC سناریوی فعال را برای نمایش در مودال می‌گیرد.
    getParameters(scenario_id) {
        return this.http
            .get(`${API_BASE_URL}/hsc_parameters/${scenario_id}`)
            .then((res) => res.data.hsc_parameters ?? res.data);
    }

    // ذخیره‌ی مقادیرِ (پیش‌فرضِ) HSC برای سناریوی فعال؛ هنگام ساخت سناریوی جدید.
    saveParameters(data) {
        return this.http({
            method: "POST",
            url: `${API_BASE_URL}/hsc_parameters`,
            data,
        }).then((res) => res.data);
    }

    // به‌روزرسانی پارامترهای HSC سناریوی فعال پس از ویرایش توسط کاربر.
    updateParameters(data, scenario_id) {
        return this.http({
            method: "PUT",
            url: `${API_BASE_URL}/hsc_parameters/${Number(scenario_id)}`,
            data,
        }).then((res) => res.data);
    }
}
