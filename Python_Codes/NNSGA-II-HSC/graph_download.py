# # -*- coding: utf-8 -*-
import os
import numpy as np
import osmnx as ox
from typing import Tuple, Optional
from pathlib import Path
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.geometry import Point, LineString
from fastapi import FastAPI, HTTPException, Body, Header
import networkx as nx

class GraphDownload():
    # --- تنظیمات عمومی OSMnx (کش، لاگ، تایم‌اوت) ---
    ox.settings.use_cache = True                     # استفاده از کش تا هر بار دانلود نشود
    ox.settings.cache_folder = "../osmnx_cache"       # پوشه کش
    ox.settings.log_console = True                   # نمایش لاگ و پیشرفت در کنسول
    ox.settings.timeout = 180                        # ثانیه
    ox.settings.nominatim_timeout = 60               # مخصوص ژئو-کدینگ
    def __init__(self):
        # ---- تنظیمات ----
        self.DATA_SHP = os.environ.get("ROADS_SHP", r"C:\Users\Amir\Desktop\SDSS-Project\Data\Roads\Tehran_Roads_line.shp")
        self.TARGET_EPSG = int(os.environ.get("TARGET_EPSG", "32639"))  # UTM zone 39N
        self.ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me")   # برای /reload
        CACHE_DIR = os.environ.get("CACHE_DIR", Path(__file__).parent / "data" / "cache")
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.CACHE_GRAPH = os.path.join(CACHE_DIR, "graph.pkl")
        self.CACHE_NODES = os.path.join(CACHE_DIR, "nodes.npy")
        self.CACHE_NODEIDS = os.path.join(CACHE_DIR, "node_ids.npy")
        self.CACHE_TRANSFORMER = os.path.join(CACHE_DIR, "transformer.pkl")
        # ---- متغیرهای سراسری ----
        self.G: Optional[nx.Graph] = None
        self.NODE_XY: Optional[np.ndarray] = None     # [[x,y], ...] in TARGET_EPSG
        self.NODE_IDS: Optional[np.ndarray] = None    # [node_id, ...]
        self.TREE: Optional[cKDTree] = None
        self.GEODETIC_TO_TARGET: Optional[Transformer] = None   # 4326 -> TARGET_EPSG
        
    def _prepare_kdtree(self, G):
        # نودها رو از lon/lat به UTM ببر
        nodes_ll = np.array([(d["x"], d["y"]) for n, d in G.nodes(data=True)], dtype=float)
        nodes_utm = np.array([self.GEODETIC_TO_TARGET.transform(lon, lat) for lon, lat in nodes_ll], dtype=float)
        node_ids = np.array(list(G.nodes()))
        tree = cKDTree(nodes_utm)
        return nodes_utm, node_ids, tree
    
    def download_tehran_district6_graph(self):
        """
        دانلود گراف خیابان‌های منطقه ۶ تهران از OSM
        - اگر قبلاً دانلود و ذخیره شده باشد، همان را لود می‌کند.
        - خروجی: گراف NetworkX
        """
        self.GEODETIC_TO_TARGET = Transformer.from_crs(4326, self.TARGET_EPSG, always_xy=True)
        # مسیر خروجی (یک پوشه عقب‌تر از این فایل → data/cache)
        base_dir = Path(__file__).parent        # مسیر پوشه Python_Codes
        out_dir = (base_dir.parent / "data" / "cache")
        out_dir.mkdir(parents=True, exist_ok=True)

        graph_path = out_dir / "tehran_m6_drive.graphml"
        # اگر فایل قبلاً وجود دارد → لود کن
        if graph_path.exists():
            print(f"[LOAD] گراف از کش لود شد → {graph_path}")
            self.G = ox.load_graphml(graph_path)
            self.NODE_XY, self.NODE_IDS, self.TREE = self._prepare_kdtree(self.G)
            return self.G

        # اگر وجود ندارد → دانلود و ذخیره
        place_candidates = [
            "District 6, Tehran, Iran",
            "Region 6, Tehran, Iran",
            "Municipal District 6, Tehran, Iran",
            "Tehran 6th District, Iran",
            "منطقه ۶ تهران, تهران, ایران",
        ]
        gdf = None
        for place in place_candidates:
            try:
                gdf = ox.geocode_to_gdf(place)
                if not gdf.empty:
                    print(f"[OK] مرز پیدا شد: {place}")
                    break
            except Exception as e:
                print(f"[WARN] '{place}' نشد → {e}")

        if gdf is None or gdf.empty:
            raise RuntimeError("نتوانستم مرز «منطقه ۶ تهران» را پیدا کنم.")
        polygon = gdf.geometry.unary_union

        # ساخت گراف
        self.G = ox.graph_from_polygon(polygon, network_type="drive", simplify=True, retain_all=False)
        self.NODE_XY, self.NODE_IDS, self.TREE = self._prepare_kdtree(self.G)
        
        # کلیپ دقیق به پلی‌گون (جایگزین clean_periphery قدیمی)
        ox.save_graphml(self.G, filepath=graph_path)
        print(f"[SAVE] گراف ذخیره شد → {graph_path}")

        return self.G

    def _nearest_node_id(self, lng, lat, tree, node_ids):
        x, y = self.GEODETIC_TO_TARGET.transform(lng, lat)
        _, idx = tree.query([x, y])
        node_id = node_ids[idx]
        return node_id, idx

    def _route(self, graph, source, target):
        u_id, _ = self._nearest_node_id(source[0], source[1], self.TREE, self.NODE_IDS)
        v_id, _ = self._nearest_node_id(target[0], target[1], self.TREE, self.NODE_IDS)
        # Dijkstra
        try:
            path = ox.shortest_path(graph, u_id, v_id, weight="length")
            route_gdf = ox.routing.route_to_gdf(graph, path)
            path_coords = []
            for idx, row in route_gdf.iterrows():
                geometry = row['geometry']
                list_coords = list(geometry.coords)
                for l in list_coords:
                    path_coords.append(l)
            path_coords = list(dict.fromkeys(path_coords))
            # GeoJSON ساده
            return {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": path_coords
                },
                "properties": {
                    "weight": float(nx.path_weight(graph, path, weight="length")),
                    "nodes": len(path)
                }
            }
        except nx.NetworkXNoPath:
            raise HTTPException(status_code=404, detail="No path found")