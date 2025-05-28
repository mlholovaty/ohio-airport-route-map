import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium
import math
import heapq
from collections import defaultdict, deque
from itertools import pairwise
import branca.colormap as cm

# -------------------------------------------------
# Load the coordinate table once at start-up
# -------------------------------------------------
airports = pd.read_csv("ohio_airports_coordinates.csv")

coords = airports.set_index("ID")[["Latitude (Decimal)",
                                   "Longitude (Decimal)"]]

KENT = coords.loc["KENT"]
st.set_page_config(layout="wide")
st.title("Ohio Airport Route Finder")

m = folium.Map(location=KENT, zoom_start=7, tiles="OpenStreetMap", control_scale=True, width='100%', height='100%') # Creates a new Map centered on Kent State University Airport

# -------------------------------------------------
# Create layers for wind velocity and dirrection
# from the National Weather Service
# -------------------------------------------------

# Add wind velocity layer
folium.raster_layers.WmsTileLayer(
    url="https://digital.weather.gov/ndfd/wms",   # master endpoint
    layers="ndfd.conus.windspd",                  # 10 m wind-speed forecast
    name="Wind speed (kts)",
    fmt="image/png",
    transparent=True,
    overlay=True,
    control=True,
    attr="NWS NDFD"                               # credit
).add_to(m)

# Add wind direction layer
folium.raster_layers.WmsTileLayer(
    url="https://digital.weather.gov/ndfd/wms",
    layers="ndfd.conus.winddir",
    name="Wind direction",
    fmt="image/png",
    transparent=True,
    overlay=True,
    control=True,
).add_to(m)

# Add wind velocity legend
wind_cmap = cm.StepColormap(
    colors=[
        "#d8d8e9",  # 0 kt  – light grey-violet
        "#f3b1f3",  # 5 kt  – light magenta
        "#ea74ff",  # 10 kt – pink-magenta
        "#a35cff",  # 15 kt – purple
        "#6175ff",  # 20 kt – blue-violet
        "#29d2ff",  # 25 kt – cyan
        "#1edd7a",  # 30 kt – green
        "#c8f000",  # 35 kt – yellow-green
        "#ffb200",  # 40 kt – orange
        "#ff5a36",  # 45 kt – red-orange
        "#b20034",  # 60 kt – deep red
    ],
    index=[0,5,10,15,20,25,30,35,40,45,60],
    vmin=0,
    vmax=60,
    caption="Wind speed (knots)"
)
wind_cmap.add_to(m)

# Add wind direction legend
dir_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]   # boundaries in degrees
dir_colors = [
    "#6e7fc0",   # N   – lavender-blue
    "#00a0c2",   # NE  – cyan-teal
    "#009b7a",   # E   – teal-green
    "#4a7c39",   # SE  – sage-green
    "#b96969",   # S   – rose
    "#8a6334",   # SW  – brown-orange
    "#6e7c32",   # W   – olive-green
    "#775c8c",   # NW  – violet-purple
]

dir_cmap = cm.StepColormap(
    colors = dir_colors,
    index  = dir_bins,
    vmin   = 0,
    vmax   = 360,
    caption = "Wind direction (° / compass)"
)
dir_cmap.add_to(m)

# -------------------------------------------------
# Create layers for VFR and IFR from the FAA
# -------------------------------------------------

# FAA VFR Sectional layer
folium.TileLayer(
    tiles="https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/"
          "services/VFR_Sectional/MapServer/tile/{z}/{y}/{x}",
    attr="FAA VFR Sectional – tiles.arcgis.com",
    name="VFR Sectional",
    overlay=True, control=True, max_zoom=12,
).add_to(m)

# FAA IFR layer
folium.TileLayer(
    tiles="https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/"
          "services/IFR_AreaLow/MapServer/tile/{z}/{y}/{x}",
    attr="FAA IFR Low – tiles.arcgis.com",
    name="IFR Low",
    overlay=True, control=True, max_zoom=13,
).add_to(m)

folium.LayerControl().add_to(m)

# -------------------------------------------------
# Creates a list of nodes
# -------------------------------------------------
nodes = {"KENT", "MFD", "TSO", "FZI", "OSU", "AOH", "I66", "22I",
         "ZZV", "I17", "LPR", "ØG6", "LUK", "HZY", "YNG", "4G5",
         "GAS", "3W2", "TOL", "EOP", "1ØG"}

graph1 = {n: [] for n in nodes} # create empty graph
graph2 = {n: [] for n in nodes} # create empty graph

# -------------------------------------------------
# Create a map with markers
# -------------------------------------------------
for b in coords.iterrows():
    if b[0] in nodes:
        lat, lon = b[1]
        folium.Marker(
            location=(lat, lon),
            popup=b[0],
            icon=folium.Icon(color="green", icon="plane")
        ).add_to(m)
    else:
        lat, lon = b[1]
        folium.Marker(
            location=(lat, lon),
            popup=b[0],
            icon=folium.Icon(color="blue", icon="plane")
        ).add_to(m)

# -------------------------------------------------
# Haversine helper
# -------------------------------------------------
EARTH_RADIUS_KM = 6_371.0088       # mean Earth radius

def haversine(lat1, lon1, lat2, lon2, *, r=EARTH_RADIUS_KM) -> float:
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)

    a = (math.sin(dφ/2)**2 +
         math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2)
    return 2 * r * math.asin(math.sqrt(a))

# -------------------------------------------------
# Compute path length + leg table
# -------------------------------------------------
def path_length(path_ids):
    hops, total = [], 0.0
    for a, b in zip(path_ids[:-1], path_ids[1:]):
        lat1, lon1 = coords.loc[a]
        lat2, lon2 = coords.loc[b]
        d_km = haversine(lat1, lon1, lat2, lon2)
        total += d_km
        hops.append((a, b, d_km))
    return total, hops

# -------------------------------------------------
# Create a graph with edges
# -------------------------------------------------
for n in nodes:
    for c in coords.iterrows():

        if n == c[0]:
            continue

        ids = c[0]
        path_ids = [n, ids]
        total_km, legs = path_length(path_ids)

        if total_km > 105: # skip long hops; there is no need to add them
            continue

        if total_km <= 52:
            graph1[n].append((ids, total_km))
        elif ids in nodes:  # if the other ID is a node
            graph2[n].append((ids, total_km))
        
        # Add specific edges based on the original code logic
        if n == "GAS" and ids == "HTW":
            graph1[n].append((ids, total_km))
        if n == "1ØG" and ids == "4I3":
            graph1[n].append((ids, total_km))
        if n == "LUK" and ids == "GEO" or n == "LUK" and ids == "OXD":
            graph1[n].append((ids, total_km))
        if n == "OSU" and ids == "38I":
            graph1[n].append((ids, total_km))

# Make sure the graph1 (<50km) is bidirectional
for u in list(graph1.keys()):                                   # iterate over a static copy of the keys
    for v, w in graph1[u]:                                      # for every edge u → v (weight w)
        back_edges = [nbr for nbr, _ in graph1.get(v, [])]
        if u not in back_edges:                                 # if v → u isn’t there yet
            graph1.setdefault(v, []).append((u, w))

# -------------------------------------------------
# Combine the two graphs
# -------------------------------------------------
def combine_keep_all(g1, g2):
    merged = defaultdict(list)

    for g in (g1, g2):           # walk both graphs
        for u, nbrs in g.items():    # every source node
            merged[u].extend(nbrs)   # tack on its edges

    return dict(merged)

graph = combine_keep_all(graph1, graph2)

for u, nbrs in list(graph.items()): # Ensures that every node is a key
    for v, _ in nbrs:
        graph.setdefault(v, [])

# -------------------------------------------------
# Dijkstra’s algorithm
# -------------------------------------------------
def dijkstra_all_paths(graph, start, goal):
    pq      = [(0, start)]
    dist    = {start: 0}            # node -> shortest-known distance
    parent  = defaultdict(list)     # node -> [all predecessors]

    while pq:                       # ❶ keep going; no early break
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                # stale entry

        for v, w in graph.get(u, []):
            nd = d + w              # tentative distance to v
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                parent[v] = [u]     # ❷ new best ⇒ reset predecessor list
                heapq.heappush(pq, (nd, v))
            elif nd == dist[v]:     # ❸ tie ⇒ *add* another predecessor
                parent[v].append(u)

    #  Build all paths from start to goal
    paths, stack = [], deque([[goal]])
    while stack:
        path = stack.pop()
        last = path[-1]
        if last == start:
            paths.append(path[::-1])
        else:
            for p in parent[last]:
                stack.append(path + [p])

    return dist.get(goal, float("inf")), paths


# -------------------------------------------------
# Handle user input for two selected points
# -------------------------------------------------

# Dropdowns to select airports
airport_ids = coords.index.tolist()
from_id = st.selectbox("Select departure airport:", airport_ids)
to_id = st.selectbox("Select destination airport:", airport_ids)

if from_id != to_id:

    if from_id not in graph or to_id not in graph:
        st.error(f"One or both of the selected airports ({from_id}, {to_id}) are not connected in the graph.")
        st.stop()
    
    distance, paths = dijkstra_all_paths(graph, from_id, to_id)
    smallest = min(paths, key=len)

    if from_id not in nodes:
        st.warning(f"The departure airport {from_id} is not a node, please check if battery is enough to reach {smallest[1]}.")
    if len(smallest) > 2:
        for start, end in pairwise(smallest):
            line_pts = [
            coords.loc[start][['Latitude (Decimal)', 'Longitude (Decimal)']].tolist(),
            coords.loc[end][['Latitude (Decimal)', 'Longitude (Decimal)']].tolist()
            ]
            folium.PolyLine(
                locations=line_pts,
                color="blue",
                weight=5,
                opacity=0.7,
            ).add_to(m)
    else:
        id_a, id_b = smallest
        folium.PolyLine(
            locations=[coords.loc[id_a], coords.loc[id_b]],
            color="green",
            weight=5,
            opacity=0.7,
        ).add_to(m)
    
    st.success(f"The total distance from {from_id} to {to_id}: **{distance:,.2f} km**"
               f" The path is: {smallest}")

    warning = "WARNING: This route is at the limit of the aircraft's range, and although it is possible to fly this route and back, \n it is not recommended. Please check the weather and power before departure."

    if "HTW" in smallest:
        st.warning(warning)
    elif "4I3" in smallest:
        st.warning(warning)
    elif "GEO" in smallest or "OXD" in smallest:
        st.warning(warning)
    elif "38I" in smallest:
        st.warning(warning)

st.text("Green markets are nodes, meaning possible rechargers, blue markers are other airports.")
data = st_folium(m, width='100%', height=700)
