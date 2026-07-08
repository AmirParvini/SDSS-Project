// services/ScenarioService.js
// -----------------------------------------------------------------------------
// لایه‌ی دسترسی به داده (Repository) برای «سناریوها». تنها مسئولیتش: صحبت با بک‌اند.
//
// نگاشت اصول SOLID:
//  - SRP: فقط CRUD و فعال‌سازی سناریو را می‌شناسد؛ هیچ منطق DOM/نقشه‌ای ندارد.
//  - DIP: axios از بیرون تزریق می‌شود (constructor injection)، پس در تست
//         می‌توان یک client جعلی جای‌گزین کرد. Controller هم به این کلاس از
//         طریق «قرارداد متدها» وابسته است، نه به axios.
//  - ISP: فقط متدهایی را که کاربرِ سناریو لازم دارد بیرون می‌دهد.
// -----------------------------------------------------------------------------
import axios from "axios";
import { API_BASE_URL } from "../config/nodeConfig.js";

export default class ScenarioService {
    constructor(httpClient = axios) {
        this.http = httpClient;
    }

    // دریافت لیست همه‌ی سناریوها + شناسه‌ی سناریوی فعالِ فعلی در دیتابیس.
    // انتظار می‌رود بک‌اند چیزی شبیه این برگرداند:
    //   { scenarios: [{ id, name, description, is_active }], active_id }
    listScenarios() {
        return this.http.get(`${API_BASE_URL}/scenarios`).then((res) => ({
            scenarios: res.data.scenarios || [],
            activeId: res.data.activeId ?? null,
        }));
    }

    // فعال‌سازی سناریوی انتخاب‌شده. مسئولیتِ «غیرفعال‌کردن سناریوی قبلی و
    // فعال‌کردن سناریوی جدید» بر عهده‌ی همین کنترلرِ بک‌اند است تا اتمی بماند.
    activateScenario(id) {
        return this.http({
            method: "PUT",
            url: `${API_BASE_URL}/select-scenario/${Number(id)}`,
        }).then((res) => res.data);
    }

    // ایجاد سناریوی جدید. طبق نیاز، بک‌اند بعد از ساخت آن را «فعال» می‌کند
    // (سناریوی قبلی غیرفعال و این سناریو فعال می‌شود).
    createScenario(data) {
        return this.http({
            method: "POST",
            url: `${API_BASE_URL}/scenarios`,
            data,
        }).then((res) => res.data);
    }

    // ویرایش ویژگی‌های سناریو (name / description). وضعیت فعال‌بودن را تغییر نمی‌دهد.
    updateScenario(data) {
        return this.http({
            method: "PUT",
            url: `${API_BASE_URL}/scenarios/${data.id}`,
            data,
        }).then((res) => res.data);
    }
}
