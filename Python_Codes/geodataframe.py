# %%
from fastapi import FastAPI, HTTPException, Body, Header
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
import gdf_to_graph as gtg

APP = FastAPI(title="Tehran Routing Service")
def _nearest_node_id(lng: float, lat: float):
    x, y = gtg.GEODETIC_TO_TARGET.transform(lng, lat)  # (lon,lat) -> UTM
    _, idx = gtg.TREE.query([x, y])
    node_id = gtg.NODE_IDS[idx]
    # اگر احیاناً نوعش ndarray شد، به tuple تبدیل کن
    if hasattr(node_id, "tolist"):
        node_id = tuple(node_id.tolist())
    elif isinstance(node_id, np.ndarray):
        node_id = tuple(node_id)
    return node_id, idx

# %%
def _route(u_id, v_id):
    # Dijkstra
    try:
        path = nx.shortest_path(gtg.G, source=u_id, target=v_id, weight="length_m")
        print(path)
        # به LineString تبدیل کن
        coords = [(gtg.G.nodes[n]["x"], gtg.G.nodes[n]["y"]) for n in path]
        line = LineString(coords)
        # GeoJSON ساده
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [(float(x), float(y)) for x, y in line.coords]
            },
            "properties": {
                "cost_seconds": float(nx.path_weight(gtg.G, path, weight="length_m")),
                "nodes": len(path)
            }
        }
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found")
# %%
#%%
if not gtg._load_cache():
    gtg._build_from_shapefile(gtg.DATA_SHP)

# %%
# nid = list(NODE_IDS)
# print(TREE)
# nid.index([515378.0 3774863.0])

# %%
u_id, _ = _nearest_node_id(51.44786687379446, 35.6976009003004)
v_id, _ = _nearest_node_id(51.44898811135682, 35.69863547929273)
feature = _route(u_id, v_id)
print(feature)