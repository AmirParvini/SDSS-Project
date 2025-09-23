import os
from typing import Tuple, Optional
from fastapi import FastAPI, HTTPException, Body, Header
from fastapi.responses import JSONResponse
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import linemerge
from scipy.spatial import cKDTree
from pyproj import Transformer
from joblib import dump, load
from tqdm import tqdm
import math
from matplotlib import pyplot as plt
from pathlib import Path

APP = FastAPI(title="Tehran Routing Service")
# ---- تنظیمات ----
DATA_SHP = os.environ.get("ROADS_SHP", r"C:\Users\Amir\Desktop\SDSS-Project\Data\Roads\Tehran_Roads_line.shp")
TARGET_EPSG = int(os.environ.get("TARGET_EPSG", "32639"))  # UTM zone 39N
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me")   # برای /reload

CACHE_DIR = os.environ.get("CACHE_DIR", Path(__file__).parent / "data" / "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_GRAPH = os.path.join(CACHE_DIR, "graph.pkl")
CACHE_NODES = os.path.join(CACHE_DIR, "nodes.npy")
CACHE_NODEIDS = os.path.join(CACHE_DIR, "node_ids.npy")
CACHE_TRANSFORMER = os.path.join(CACHE_DIR, "transformer.pkl")

# ---- متغیرهای سراسری ----
G: Optional[nx.Graph] = None
NODE_XY: Optional[np.ndarray] = None     # [[x,y], ...] in TARGET_EPSG
NODE_IDS: Optional[np.ndarray] = None    # [node_id, ...]
TREE: Optional[cKDTree] = None
GEODETIC_TO_TARGET: Optional[Transformer] = None   # 4326 -> TARGET_EPSG

def _load_shapefile(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise RuntimeError("Shapefile has no CRS. Add a .prj or set crs before.")
    gdf = gdf.to_crs(TARGET_EPSG)
    # پاکسازی هندسه‌های خالی
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    # Optional: explode multiline to singleparts for cleaner graph
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    # print(gdf.geometry[2])
    return gdf

def _gdf_to_graph(
    gdf: gpd.GeoDataFrame,
    use_tqdm: bool = True,
    progress_cb=None,          # تابع اختیاری: progress_cb(done, total)
    progress_every: int = 500  # هر چند رکورد یک بار کال‌بک صدا بخورد
) -> nx.Graph:
    """
    از LineStringها گراف می‌سازد و پیشرفت را با tqdm یا کال‌بک گزارش می‌دهد.
    """
    G = nx.Graph()

    total = len(gdf.geometry)
    iterator = enumerate(gdf.geometry)

    # اگر tqdm نصب است و فعال خواستی، بپیچ داخل tqdm
    if use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="Building road graph", unit="feat")

    for idx, geom in iterator:
        if geom is None or geom.is_empty:
            # گزارش پیشرفت با کال‌بک (اگر tqdm نداریم/نمی‌خواهیم)
            if progress_cb and (not use_tqdm or tqdm is None) and (idx % progress_every == 0):
                progress_cb(idx, total)
            continue

        if not isinstance(geom, LineString):
            # اگر MultiLineString باقی مانده، سعی در merge
            try:
                geom = linemerge(geom)
                if not isinstance(geom, LineString):
                    if progress_cb and (not use_tqdm or tqdm is None) and (idx % progress_every == 0):
                        progress_cb(idx, total)
                    continue
            except Exception:
                if progress_cb and (not use_tqdm or tqdm is None) and (idx % progress_every == 0):
                    progress_cb(idx, total)
                continue
            
        coords = list(geom.coords)
        for i, _ in enumerate(coords):
            if i == len(coords) - 1:
                break
            start = coords[i]; end = coords[i+1]
            u = (round(start[0], 3), round(start[1], 3))
            v = (round(end[0], 3), round(end[1], 3))

            length_m = float(math.sqrt((v[0] - u[0])**2 + (v[1] - u[1])**2))
            # speed_kmh = default_speed_kmh
            # اگر فیلد کلاس/سرعت داری، اینجا مقدار بده:
            # if 'class' in gdf.columns:
            #     cls = gdf.at[idx, 'class']
            #     speed_kmh = 90 if cls in ('motorway','trunk') else 60 if cls in ('primary','secondary') else 30

            if u not in G: G.add_node(u, x=u[0], y=u[1])
            if v not in G: G.add_node(v, x=v[0], y=v[1])

            G.add_edge(u, v, length_m=length_m, geometry=LineString([u, v]))

            # اگر tqdm نداریم، هر progress_every رکورد یک‌بار کال‌بک بزن
            if progress_cb and (not use_tqdm or tqdm is None) and (idx % progress_every == 0):
                progress_cb(idx, total)

    # مرحلهٔ وزن‌دهی یال‌ها (خارج از حلقه، سریع است)
    # _add_costs_to_edges(G)

    # در پایان، 100% را اعلام کن
    if progress_cb and (not use_tqdm or tqdm is None):
        progress_cb(total, total)
    return G

def _prepare_kdtree(G: nx.Graph):
    nodes = np.array([(d["x"], d["y"]) for n, d in G.nodes(data=True)], dtype=float)
    node_ids = np.array(list(G.nodes()))
    tree = cKDTree(nodes)
    return nodes, node_ids, tree

def _save_cache():
    dump(G, CACHE_GRAPH)
    np.save(CACHE_NODES, NODE_XY)
    np.save(CACHE_NODEIDS, NODE_IDS, allow_pickle=True)
    dump(GEODETIC_TO_TARGET, CACHE_TRANSFORMER)

def _load_cache() -> bool:
    global G, NODE_XY, NODE_IDS, TREE, GEODETIC_TO_TARGET
    if not (os.path.exists(CACHE_GRAPH) and os.path.exists(CACHE_NODES)
            and os.path.exists(CACHE_NODEIDS) and os.path.exists(CACHE_TRANSFORMER)):
        return False
    try:
        print("Loading graph cache...")
        G = load(CACHE_GRAPH)
        NODE_XY = np.load(CACHE_NODES)
        print("Loading CACHE_NODES cache...")
        NODE_IDS = np.load(CACHE_NODEIDS, allow_pickle=True)
        print("Loading CACHE_NODEIDS cache...")
        TREE = cKDTree(NODE_XY)
        print("Loading NODE_XY cache...")
        GEODETIC_TO_TARGET = load(CACHE_TRANSFORMER)
        print("Loading GEODETIC_TO_TARGET cache...")
        return True
    except (EOFError, OSError, ValueError) as e:
        print(f"Error loading cache: {e}")
        return False

def _build_from_shapefile(path: str):
    global G, NODE_XY, NODE_IDS, TREE, GEODETIC_TO_TARGET
    GEODETIC_TO_TARGET = Transformer.from_crs(4326, TARGET_EPSG, always_xy=True)
    gdf = _load_shapefile(path)
    G = _gdf_to_graph(gdf)
    NODE_XY, NODE_IDS, TREE = _prepare_kdtree(G)
    _save_cache()