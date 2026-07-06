// services/NodeApiService.js
// -----------------------------------------------------------------------------
// لایه‌ی دسترسی به داده (Repository). تنها مسئولیتش: صحبت با بک‌اند.
// اصل SRP: هیچ‌جای دیگری از برنامه نباید مستقیم axios صدا بزند.
// اصل DIP: axios از بیرون تزریق می‌شود (constructor injection)، پس
//         در تست می‌توان یک client جعلی جای‌گزین کرد.
// -----------------------------------------------------------------------------
import axios from "axios";
import { API_BASE_URL } from "../config/nodeConfig.js";

export default class NodeApiService {
    constructor(httpClient = axios) {
        this.http = httpClient;
    }

    // بارگذاری اولیه‌ی نودها و پارامترهای HSC هنگام ورود به برنامه.
    loadData() {
        return this.http.get(`${API_BASE_URL}/load_data`).then((res) => ({
            geojsonNodes: res.data.geojson_nodes,
            hscParameters: res.data.hsc_parameters,
        }));
    }

    // ایجاد یا ویرایش یک نود. وجود id تعیین می‌کند POST است یا PUT.
    updateNode(data) {
        return this.http({
            method: "PUT",
            url:`/nodes/${data.id}`,
            data,
        }).then((res) => res.data);
    }

    createNode(data) {
        return this.http({
            method: "POST",
            url:"/nodes",
            data,
        }).then((res) => res.data);
    }

    // حذف یک نود بر اساس id.
    deleteNode(id) {
        return this.http.delete(`/nodes/${id}`).then((res) => res.data);
    }
}
