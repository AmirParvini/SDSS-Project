# Route Calculation Service

محاسبه‌ی فاصله و جئومتری مسیر بین نقاط، بازنویسی‌شده بر اساس اصول SOLID.

## ساختار (هر کلاس یک مسئولیت)

| فایل | مسئولیت | اصل SOLID |
|------|---------|-----------|
| `config.py` | نگه‌داری تنظیمات (EPSG، مسیر کش، محدوده) | SRP |
| `dtos.py` | قرارداد ورودی/خروجی (GeoPoint, RoutingRequestItem, RouteResult) | SRP |
| `interfaces.py` | پورت‌های انتزاعی (Projector, GraphProvider, NodeLocator, RoutePlanner) | DIP + ISP |
| `projection.py` | تبدیل مختصات 4326 → متریک (pyproj) | SRP |
| `graph_repository.py` | گرفتن/کش گراف OSM (حافظه + دیسک) | SRP + OCP |
| `node_locator.py` | نزدیک‌ترین نود با KD-Tree | SRP |
| `geojson.py` | ساخت جئومتری استاندارد GeoJSON | SRP |
| `route_planner.py` | کوتاه‌ترین مسیر (Dijkstra) + فاصله | SRP + LSP |
| `routing_service.py` | هماهنگ‌کننده (input → output) | SRP + DIP |
| `container.py` | Composition root + singleton (گراف فقط یک‌بار ساخته می‌شود) | DIP |
| `api.py` | ورودی HTTP (FastAPI) — پیشنهادی برای Laravel | — |
| `cli.py` | ورودی CLI (stdin/stdout) — جایگزین | — |

## قرارداد ورودی/خروجی

```jsonc
// input
[{"source": {"id": 1, "lat": 35.70, "lng": 51.40},
  "targets": [{"id": 2, "lat": 35.71, "lng": 51.41}]}]

// output
[{"source_id": 1, "target_id": 2, "distance": 1234.5,
  "geometry": {"type": "LineString", "coordinates": [[lng, lat], ...]}}]
```

`distance` بر حسب متر است (اگر مسیری نباشد `null`). `geometry` یک شیء
**geometry استاندارد GeoJSON** است (نه Feature)، با ترتیب `[lng, lat]`.

## کش گراف
گراف در سه لایه کش می‌شود تا هر بار دانلود نشود:
1. **حافظه**: پس از اولین ساخت، همان نمونه تا پایان عمر پروسه استفاده می‌شود.
2. **دیسک**: فایل GraphML در `data/cache/graph.graphml`.
3. فقط اگر هیچ‌کدام نبود، از OSM دانلود و سپس ذخیره می‌شود.

## اجرا
```bash
pip install -r requirements.txt
# سرویس HTTP (پیشنهادی — گراف در حافظه می‌ماند):
uvicorn api:app --host 127.0.0.1 --port 8008
# یا CLI:
echo '[{"source":{"id":1,"lat":35.7,"lng":51.4},"targets":[{"id":2,"lat":35.71,"lng":51.41}]}]' | python cli.py
```
