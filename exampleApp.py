import streamlit as st
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.express.colors import sample_colorscale
import shapely.geometry
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon 
import json
import numpy as np
from code_editor import code_editor
from io import StringIO
import math
import datetime
import os
from pymongo import MongoClient
from datetime import timezone

def plotEllipseTissot(ra, dec, radius=20):
    theta = np.deg2rad(dec)
    phi = np.deg2rad(ra - 360 if ra > 180 else ra)
    ellipse = Ellipse((phi,theta), 2*np.deg2rad(radius)/ np.cos(theta),
                      2*np.deg2rad(radius))
    vertices = ellipse.get_verts()     # get the vertices from the ellipse object
    
    verticesDeg = np.rad2deg(vertices)
    
    ra_out = [i + 360 if i < 0  else i for i in verticesDeg[:,0]]
    dec_out = verticesDeg[:,1]

    return np.column_stack((ra_out, dec_out))
# new helper: convert axis-aligned bounds to rectangle corner coordinates
def rect_corners(xmin, xmax, ymin, ymax, closed=False):
    """
    Return corner coordinates for an axis-aligned rectangle defined by bounds.
    Parameters:
      xmin, xmax, ymin, ymax : numeric
      closed (bool) : if True, repeat the first corner at the end (useful for plotting closed polygons)
    Returns:
      numpy.ndarray shape (4,2) or (5,2) if closed: [[x1,y1], [x2,y2], [x3,y3], [x4,y4], ...]
    Order of corners: (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)
    """
    corners = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ], dtype=float)
    if closed:
        return np.vstack([corners, corners[0]])
    return corners

# --- Added: helpers to map t_frac in [0,1] to colors (red -> blue) ---
def _tfrac_to_rgb(t):
    """
    Map t_frac in [0,1] to an RGB tuple using matplotlib's 'Spectral' colormap.
    Returns (r,g,b) as 0-255 ints.
    """
    import matplotlib.cm as _cm
    t = float(np.clip(t, 0.0, 1.0))
    cmap = matplotlib.colormaps['Spectral']
    rgba = cmap(t)  # (r,g,b,a) floats in 0..1
    rgb = tuple(int(round(255 * v)) for v in rgba[:3])
    return rgb

def _rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def _rgba_str(rgb, alpha=0.25):
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'

# --- Modified: plotPolygons now draws a thin white outline under each trace ---
def plotPolygons(data, survey_id, allColours=True):


    for i in data["year1Areas"]:
        # determine target figure from top-level 'year' in data
        #print('Data being plotted:', i)
        try:
            year_val = int(i['year'])
            #print("Detected year:", year_val)
        except Exception:
            #print("Error detecting year.")
            year_val = None

        # default routing: year==2 -> fig2, otherwise -> fig1
        try:
            if year_val == 5:
                target_fig = fig5
            elif year_val == 4:
                target_fig = fig4
            elif year_val == 3:
                target_fig = fig3
            elif year_val == 2:
                target_fig = fig2
            else:
                target_fig = fig1
        except NameError:
            # if figs are not yet created, fallback to original fig
            print("Figure objects not found; defaulting to 'fig'.")
            target_fig = globals().get('fig', None)

        # compute color from t_frac (default to 0 if missing)
        tfrac = i.get('t_frac', 0.0)
        rgb = _tfrac_to_rgb(tfrac)
        hexcol = _rgb_to_hex(rgb)
        fillcol = _rgba_str(rgb, alpha=0.22)

        # outline settings (white thin line under the colored line)
        outline_color = "#ffffff"
        outline_width = 3  # slightly larger than inner line so white shows as an outline
        inner_width = 2

        if i['type']=='stripe':
            RA_lower = i['RA_lower']; RA_upper = i['RA_upper']
            Dec_lower = i['Dec_lower']; Dec_upper = i['Dec_upper']
            # compute RA span, handle wrap-around across 360->0
            if RA_upper >= RA_lower:
                ra_span = RA_upper - RA_lower
            else:
                ra_span = (RA_upper + 360.0) - RA_lower
            dec_span = abs(Dec_upper - Dec_lower)
            # enforce minimum span of 2.5 degrees for both axes
            min_span = 2.5
            if dec_span < min_span:
                st.warning(
                    f"Stripe '{i.get('name','')}' for survey {survey_id} is too small: "
                    f"Dec span={dec_span:.2f}. Minimum is {min_span}.\n This will not be plotted and is not a valid LTS input."
                )
                continue
            # build rectangle corners and plot on chosen figure
            corners = rect_corners(RA_lower, RA_upper, Dec_lower, Dec_upper, closed=True)

            # white outline trace (no fill)
            target_fig.add_trace(go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                showlegend=False,
                mode="lines",
                line=dict(color=outline_color, width=outline_width),
                hoverinfo='skip',
            ))

            # colored filled trace on top
            target_fig.add_trace(go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                showlegend=False,
                mode="lines",
                fill="toself",
                line=dict(color=hexcol, width=inner_width),
                fillcolor=fillcol,
                name=f"{survey_id}<br> t_frac: {tfrac}"
            )
                        )
        elif i['type']=='point':
            ra_center = i['RA_center']
            dec_center = i['Dec_center']
            radius = i.get('radius', 1.15)
            tfrac = i['t_frac']
            tissot = plotEllipseTissot(ra_center, dec_center, radius = radius)

            # white outline trace (no fill)
            target_fig.add_trace(go.Scatter(
                x=tissot[:, 0],
                y=tissot[:, 1],
                showlegend=False,
                mode="lines",
                line=dict(color=outline_color, width=outline_width),
                hoverinfo='skip',
            ))

            # colored filled trace on top
            target_fig.add_trace(go.Scatter(
                x=tissot[:, 0],
                y=tissot[:, 1],
                showlegend=False,
                mode="lines",
                fill="toself",
                line=dict(color=hexcol, width=inner_width),
                fillcolor=fillcol,
                name=f"{survey_id}<br> t_frac: {tfrac}"
            )
                        )

        elif i['type']=='polygon' or i['type']=='box':
            RA = i['RA']
            Dec = i['Dec']
            tfrac = i['t_frac']
            convex_hull = np.array(
                shapely.geometry.MultiPoint(
                    [xy for xy in zip(RA, Dec)]
                ).convex_hull.exterior.coords
            )

            # white outline trace (no fill)
            target_fig.add_trace(go.Scatter(
                x=convex_hull[:, 0],
                y=convex_hull[:, 1],
                showlegend=False,
                mode="lines",
                line=dict(color=outline_color, width=outline_width),
                hoverinfo='skip',
            ))

            # colored filled trace on top
            target_fig.add_trace(go.Scatter(
                x=convex_hull[:, 0],
                y=convex_hull[:, 1],
                showlegend=False,
                mode="lines",
                fill="toself",
                line=dict(color=hexcol, width=inner_width),
                fillcolor=fillcol,
                name="t_frac: "+str(tfrac)
            )
                        )
        
        else:
            print("Please enter a valid shape: 'stripe', 'point', or 'polygon' (or 'box').")
            continue


def computeTimePressures(data):
    """
    Build a weight map (same shape as grid_map_nan) from the provided submission data.
    Returns zeros map if data missing or malformed.
    """
    # guard against missing data
    if not data or 'year1Areas' not in data or not isinstance(data['year1Areas'], list):
        return np.zeros_like(grid_map_nan)

    from matplotlib.path import Path

    truthGrids = []
    for i in data["year1Areas"]:
        try:
            if i.get('type') == 'stripe':
                RA_lower = i['RA_lower']; RA_upper = i['RA_upper']
                Dec_lower = i['Dec_lower']; Dec_upper = i['Dec_upper']
                tfrac = float(i.get('t_frac', 0.0))
                convex_hull = rect_corners(RA_lower, RA_upper, Dec_lower, Dec_upper, closed=True)

            elif i.get('type') == 'point':
                ra_center = i['RA_center']
                dec_center = i['Dec_center']
                radius = i.get('radius', 1.15)
                tfrac = float(i.get('t_frac', 0.0))
                tissot = plotEllipseTissot(ra_center, dec_center, radius=radius)
                convex_hull = tissot

            elif i.get('type') == 'polygon' or i.get('type') == 'box':
                RA = i['RA']
                Dec = i['Dec']
                tfrac = float(i.get('t_frac', 0.0))
                convex_hull = np.array(
                    shapely.geometry.MultiPoint(
                        [xy for xy in zip(RA, Dec)]
                    ).convex_hull.exterior.coords
                )
            else:
                # skip unknown types
                continue

            # flatten mesh -> Nx2 array of (lon, lat)
            allPoints = np.vstack(list(map(np.ravel, mesh))).T  # shape (N,2)

            # Use matplotlib Path for point-in-polygon testing (robust & fast)
            path = Path(convex_hull)
            inShape = path.contains_points(allPoints)
            weightMapFlat = inShape.astype(float) * tfrac
            weightMap = np.reshape(weightMapFlat, grid_map_nan.shape)
            truthGrids.append(weightMap)
        except Exception:
            # on any area-specific error skip that area
            continue

    if not truthGrids:
        return np.zeros_like(grid_map_nan)

    truthGrid = np.maximum.reduce(truthGrids)
    return truthGrid

def moving_average_2d_wrap(arr, width):
    # width must be odd so the window is centered
    assert width % 2 == 1, "width must be odd"

    k = width // 2
    result = np.zeros_like(arr, dtype=float)

    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            result += np.roll(np.roll(arr, dx, axis=0), dy, axis=0)

    return result / (width * width)

# new helper: 1D circular moving average (for longitude series)
def moving_average_1d_wrap(arr, width):
    """
    Circular moving average over 1D array.
    width must be odd. Returns array same shape as input.
    """
    arr = np.asarray(arr, dtype=float)
    assert width % 2 == 1, "width must be odd"
    k = width // 2
    # use rolling sum via np.roll
    result = np.zeros_like(arr, dtype=float)
    for shift in range(-k, k+1):
        result += np.roll(arr, shift)
    return result / width


f = open('demoArea.json')
demo_io = f.read()

xsize = 420
ysize = xsize/2
longitude = np.linspace(0,360, int(xsize))
latitude = np.linspace(-90, 90, int(ysize))
mesh = np.meshgrid(longitude, latitude)
grid_map_nan = np.load('ltsVPSelfie453.npy')

zmin = 10
zmax = np.nanmax(grid_map_nan)




dataDefault = json.loads(str(demo_io))




# new: helpers to parse ISO timestamps and fetch latest submission per survey
def _parse_iso_ts(ts):
    import datetime as _dt
    if isinstance(ts, _dt.datetime):
        return ts
    try:
        # convert trailing 'Z' to +00:00 so fromisoformat accepts it
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def get_latest_submissions_by_survey(mongo_uri, db_name="lts", coll_name="year1submissions"):
    """
    Return dict: { survey_id: { "data": <doc.data>, "timestamp": <doc.timestamp>, "filename": <doc.filename>, "_id": <doc._id> } }
    where the document chosen is the latest (by timestamp) for that survey.
    """
    client = None
    latest = {}
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # force connect
        coll = client[db_name][coll_name]
        for doc in coll.find({}):
            survey = None
            try:
                survey = doc.get("data", {}).get("survey")
            except Exception:
                survey = None
            if not survey:
                continue
            ts = _parse_iso_ts(doc.get("timestamp"))
            if ts is None:
                continue
            cur = latest.get(survey)
            if cur is None or ts > cur["_parsed_ts"]:
                # store parsed timestamp for comparison and keep full doc
                latest[survey] = {"_parsed_ts": ts, "doc": doc}
        # build return mapping that exposes the full "data" column plus metadata
        result = {}
        for s, entry in latest.items():
            d = entry["doc"]
            result[s] = {
                "data": d.get("data"),
                "timestamp": d.get("timestamp"),
                "filename": d.get("filename"),
                "_id": d.get("_id")
            }
        return result
    except Exception as e:
        # on error return empty dict (caller may show message)
        return {}
    finally:
        try:
            if client:
                client.close()
        except Exception:
            pass

st.set_page_config(layout="wide")

# query MongoDB for latest submissions per survey (if secrets provided)
try:
    mongo_uri = st.secrets.get("MONGO_URI")
    mongo_db = st.secrets.get("MONGO_DB", "lts")
    mongo_coll = st.secrets.get("MONGO_COLLECTION", "year1all")
except Exception:
    mongo_uri = None
    mongo_db = "lts"
    mongo_coll = "year1all"

latest_submissions = {}
if mongo_uri:
    latest_submissions = get_latest_submissions_by_survey(mongo_uri, mongo_db, mongo_coll)

# convert latest_submissions to a JSON string for download/display (ObjectId/datetime -> str)
try:
    latest_submissions_json = json.dumps(latest_submissions, indent=4, default=str)
except Exception:
    print("Error converting latest submissions to JSON")
    latest_submissions_json = "{}"

# --- Added: prefer latest submission for survey S00 if present (non-empty), else use demo_data ---
# dataDefault is the fallback used when the editor content fails to parse
try:
    s00_entry = latest_submissions.get("S00") if latest_submissions else None
except Exception:
    s00_entry = None

if s00_entry and s00_entry.get("data"):
    #print('using latest S00 submission as default')
    dataDefault = json.dumps(s00_entry["data"], indent=4, default=str)
    #print('Default Data S00', dataDefault)
else:
    dataDefault = json.loads(str(demo_io))

#print(latest_submissions)
# show a compact summary in the sidebar
# if latest_submissions:
#     try:
#         # display full contents for each survey
#         st.sidebar.header("Latest submissions by survey")
#         st.sidebar.json(latest_submissions)
#     except Exception:
#         pass

st.title("PSOC LTS Tool")

# st.write("""This web tool allows 4MOST surveys to define their Year 1 sky observation preferences as part of the Long-Term Scheduler development efforts.
# """)

# st.divider()
# st.header("Step 1: Define Areas here")

# st.write("""
# Only stripes and single pointings are accepted into the 4MOST Year 1 Long Term Scheduler V2.
         
# Edit the JSON contents below, or paste in your own code. 
         
# Click the :grey-background[:orange[Run \u25BA]] button. when ready to view submission.
# Both the Sky Plot and R.A. Pressure will update accordingly.
# """)

# st.markdown("""
# Edit the value for the `survey` key with your surveys ID. e.g. S01, S02, etc. Expected format: string.

# **Important:** Enter your science justification for this request in the `scienceJustification` part. In the event of oversubscription in an R.A. range, the science justification is a useful negotiation tool.
#  Do not use the \'\"\' character as this will break you out of the string.

# Only acceptable inputs for polygons are: 'stripe' and 'point'. Examples are shown at the end of the page.

# #### t_frac
# The `t_frac` key is common to all polygons. It is where you define what fraction of the total 5-year observing time you would like to use in Year 1. It should take a value between 0-1.
            
# Example Polygons are shown at the bottom of the page.  
# """)

#{"name": "Copy", "hasText":True, "alwaysOn": True,"style": {"top": "0.46rem", "right": "0.4rem", "commands": ["copyAll"]}}
custom_btns = [{"name": "Run",
"feather": "Play",
"primary": True,
"alwaysOn":True,
"hasText": True,
"showWithIcon": True,
"commands": ["submit"],
"style": {"bottom": "0.44rem", "right": "0.4rem"}
}]
#print('Default Data before response', dataDefault)
response_dict = code_editor(str(dataDefault), lang="json", buttons=custom_btns, height=[10, 20])
#print('Response Dict text', response_dict['text'])
# Robust JSON parsing: show parse error to user and fall back to dataDefault
try:
    raw_text = str(response_dict['text'])
    if not raw_text.strip():
        data = json.loads(str(dataDefault))
    else:
        data = json.loads(raw_text)
    #print('Parsed Data', data)
except Exception as e:
    st.error(f"Error parsing editor JSON: {e}")
    st.info("Using fallback submission data (demo or latest S00) until JSON is valid.")
    data = dataDefault

# compute time pressures for the currently parsed data so plots update immediately
try:
    truthGridCurrent = computeTimePressures(data)
except Exception as e:
    # avoid breaking the app if compute fails; show a message and use zeros
    st.error(f"Error computing time pressures: {e}")
    truthGridCurrent = np.zeros_like(grid_map_nan)

# numColours = np.linspace(0, 1, len(data["year1Areas"])+1)
# colours = iter(sample_colorscale('Tealgrn', list(numColours)))

def colorbar(zmin, zmax, n = 6):
    return dict(
        title = "Total Exposure Time<br>in pixel (minutes)",
        tickmode = "array",
        tickvals = np.linspace(np.log10(zmin), np.log10(zmax), n),
        ticktext = np.round(10 ** np.linspace(np.log10(zmin), np.log10(zmax), n), 0)
    )





layout = go.Layout(
    autosize=False,
    width=800, 
    height=600,
    title='Year 1 Long Term Scheduler Preference: SELFIE 453',
    xaxis=dict(
        title='R.A.',

    ),
    yaxis=dict(
        title='Declination',

    ))


fig = go.Figure(go.Heatmap(
        x=longitude,
        y=latitude,
        z=np.ma.log10(grid_map_nan),
    text=grid_map_nan,
    hovertemplate = 
    "<i>4MOST VP Exposure Time</i><br>" +
    "<b>RA</b>: %{x}<br>" +
    "<b>Decl.</b>: %{y}<br>" +
    "<b>Total t_exp (min)</b>: %{text:.1f}",
    zmin = np.log10(zmin), zmax = np.log10(zmax),
    colorbar = colorbar(zmin, zmax, 12),
    colorscale = 'Viridis',
    name=""
    ), layout=layout)

# create fig1..fig5 immediately so plotPolygons can route to them
old_title = fig.layout.title.text if (hasattr(fig.layout, "title") and fig.layout.title and getattr(fig.layout.title, "text", None)) else "Year 1 Long Term Scheduler Preference"
figs = []
for idx in range(1, 6):
    newf = go.Figure(fig)  # copy figure (data + layout)
    newf.update_layout(title=f"Year {idx}",yaxis_range=[-90,30])
    newf['layout']['xaxis']['autorange'] = "reversed"
    figs.append(newf)
fig1, fig2, fig3, fig4, fig5 = figs

if latest_submissions:
    for i in latest_submissions.keys():
        dataLatest = latest_submissions[i]['data']
        plotPolygons(dataLatest, dataLatest.get('survey', i), allColours=False)
else:
    st.info("No previous submissions found in the remote DB — only the current edited data will be plotted.")
#print('Data', data)

plotPolygons(data, data['survey'], allColours=True)

if latest_submissions:
    latestTPress = []
    for i in latest_submissions.keys():
        dataLatest = latest_submissions[i]['data']
        latestTPress.append(computeTimePressures(dataLatest))
    truthGridLatest = np.maximum.reduce(latestTPress)
try:
    #print(truthGridLatest)
    truthGridLatest = np.maximum.reduce([truthGridCurrent, truthGridLatest])
    scaledGrid = (truthGridLatest) * grid_map_nan


    time5year = np.nansum(grid_map_nan, axis=0)
    timeMax1year = time5year/5.0
    timeY1 = np.nansum(scaledGrid, axis=0)

    widthWant = len(longitude)/360
    binsWant = 30//widthWant
    coarseTime = timeY1/timeMax1year
    smoothTime = moving_average_2d_wrap(coarseTime, width=25)
    #print(smoothTime)
    plotSmooth = True
except:
    plotSmooth = False

fig['layout']['xaxis']['autorange'] = "reversed"
fig.update_layout(yaxis_range=[-90,30])

st.divider()
st.header("Step 2: Check output on sky map")
st.markdown("""
Inspect the sky map here before moving on to the submission step.

The goal is to avoid oversubscription in Year 1 any R.A. range, which is indicated by the R.A. Time Pressure plot below the sky map.
We do not want to spend more than 50% of the available time in any R.A. range.
R.A. pressure is smoothed over a rolling 30 degrees width.
""")

# display the five pre-created figures vertically
# --- Added: horizontal Spectral colorbar displayed above the stacked figures ---
# build a Plotly colorscale sampled from matplotlib Spectral
ncolors = 256
cmap = matplotlib.colormaps['Spectral']
colors = [_rgb_to_hex(tuple(int(round(255 * v)) for v in cmap(i / (ncolors - 1))[:3])) for i in range(ncolors)]
colorscale = [[i / (ncolors - 1), colors[i]] for i in range(ncolors)]

# --- Replace the previous colorbar_fig with a thinner colorbar that shows inner ticks 0..1 in 0.1 increments ---
z = np.linspace(0.0, 1.0, ncolors).reshape(1, -1)
# build tick positions and labels 0.0, 0.1, ... 1.0
tick_vals = [round(i * 0.1, 1) for i in range(11)]
tick_text = [f"{v:.1f}" for v in tick_vals]

colorbar_fig = go.Figure(go.Heatmap(
    z=z,
    colorscale=colorscale,
    showscale=True,
    colorbar=dict(
        orientation='h',
        thickness=6,              # thinner bar body
        len=1,                   # fill the full x width
        x=0.5,
        xanchor='center',  
        tickmode='array',
        tickvals=tick_vals,      # inner ticks at 0.0..1.0 (0.1 steps)
        ticktext=tick_text,
        ticks='inside',          # draw ticks inside the bar
        tickfont=dict(size=10)
    )
))

# compact figure but allow top margin so title/ticks aren't clipped
colorbar_fig.update_layout(
    height=60,
    margin=dict(l=8, r=8, t=28, b=6),
    xaxis=dict(visible=False, range=[-0.5, ncolors - 0.5]),
    yaxis=dict(visible=False),
)

st.plotly_chart(colorbar_fig)

# display the five pre-created figures vertically
for f in (fig1, fig2, fig3, fig4, fig5):
    st.plotly_chart(f)

# New: 1D line plot under the map that shares the longitude x-axis scale
#import plotly.graph_objects as go as _go  # avoid name clash in context; use existing go normally
# if plotSmooth:
#     fig_times = go.Figure()
#     fig_times.add_trace(go.Scatter(
#         x=longitude,
#         y=coarseTime,
#         mode="lines",
#         name="Coarse Bins",
#         line=dict(color="#b1b1b1", width=2, dash='dash')
#     ))
#     fig_times.add_trace(go.Scatter(
#         x=longitude,
#         y=smoothTime,
#         mode="lines",
#         name="30° Smooth",
#         line=dict(color="#96cefd", width=5)
#     ))
#     fig_times.add_hline(y=0.5, line_width=2, line_dash="dash", line_color="#72e06a", annotation_text="50% Time Pressure", annotation_position="bottom left")
#     fig_times.add_hline(y=0.8, line_width=2, line_dash="dash", line_color="#d31510", annotation_text="80% Time Pressure", annotation_position="bottom left")
#     fig_times.update_layout(
#         autosize=False,
#         width=800,
#         height=260,
#         title="R.A. Time Pressure Plot",
#         xaxis=dict(title="R.A.", range=[360, 0]),  # reversed to match sky map RA direction
#         yaxis=dict(title="Fraction of 1-year time", range=[0, 1]),
#         margin=dict(l=40, r=20, t=50, b=40),
#     )
#     st.plotly_chart(fig_times)

st.divider()
st.header("Step 3: Save to cloud")
st.markdown("""
The JSON file contents in the editor will be saved to a remote database.
            
Select your survey from the dropdown list, click the download button.
""")
sb = st.columns((1,9))
surveyNumber=None
surveyNumber = sb[0].selectbox(
    'Select Survey',
    ('01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18'),
    index=None,
    placeholder="S00")
if surveyNumber == None:
    surveyNumber = '00'
today = datetime.date.today()
fileOutputName = 'S'+str(surveyNumber)+'_'+'LTSYear1'+'_'+str(today.year)+today.strftime('%m')+today.strftime('%d')+'.json'
st.write('File name:', fileOutputName)
json_string = json.dumps(data,indent=4, separators=(',', ': '))

def save_to_remote_db(json_text, filename):
    """
    Save JSON to a MongoDB collection.
    Expects environment variables (or defaults):
      MONGO_URI          - MongoDB connection string (required)
      MONGO_DB           - database name (default: 'lts')
      MONGO_COLLECTION   - collection name (default: 'year1submissions')
    The inserted document will have fields:
      filename, timestamp (UTC ISO), data (parsed JSON)
    """
    mongo_uri = st.secrets["MONGO_URI"]
    if not mongo_uri:
        return False, "MONGO_URI environment variable not set."

    db_name = st.secrets["MONGO_DB"]
    coll_name = st.secrets["MONGO_COLLECTION"]

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # trigger connection check
        client.server_info()

        db = client[db_name]
        coll = db[coll_name]

        doc = {
            "filename": filename,
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "data": json.loads(json_text)
        }

        result = coll.insert_one(doc)
        client.close()
        return True, f"Saved to MongoDB (id: {result.inserted_id})."
    except Exception as e:
        try:
            client.close()
        except:
            pass
        return False, f"Error saving to MongoDB: {e}"

# Replace single download button with side-by-side download + save
# ...existing code...
# Inject CSS: disabled buttons grey, enabled buttons green/strong
st.markdown(
    """
    <style>
    /* Disabled buttons (greyed out) */
    .stButton>button[disabled] {
        background-color: #6c757d !important; /* grey */
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 0.6 !important;
        border: none !important;
    }
    /* Enabled buttons (emphasised, green) */
    .stButton>button:not([disabled]) {
        background-color: #28a745 !important; /* green */
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 1.2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 18px rgba(40,167,69,0.25) !important;
        border: none !important;
    }
    .stButton>button:not([disabled]):hover {
        background-color: #218838 !important;
    }
    .stButton {
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.download_button(
    label="Download JSON File (optional)",
    data=json_string,
    file_name=fileOutputName,
    mime="application/json",
)

st.markdown("### Save to remote database", unsafe_allow_html=True)
# password input
pw = st.text_input("Upload password (press enter to confirm)", type="password", key="upload_pw_input", help="Enter password to enable Save")
# expected password from secrets
expected_pw = None
try:
    expected_pw = st.secrets.get("UPLOAD_PASSWORD")
except Exception:
    expected_pw = None

if expected_pw is None:
    st.warning("Upload password not configured in secrets. Saving is disabled.")
    save_disabled = True
else:
    save_disabled = (pw != expected_pw)

if pw and expected_pw and pw != expected_pw:
    st.error("Password incorrect.")

# Save button disabled until correct password entered
if st.button("Save to remote DB", key="save_remote_db", disabled=save_disabled):
    ok, msg = save_to_remote_db(json_string, fileOutputName)
    if ok:
        st.success('You must refresh the page before making more edits.\n' + msg)
    else:
        st.error(msg)

st.divider()
st.header("Latest submissions")
if not latest_submissions:
    st.info("No submissions found in the remote DB.")
else:
    rows = []
    for s in sorted(latest_submissions.keys()):
        ent = latest_submissions[s]
        data_field = ent.get("data", {}) or {}
        justification = data_field.get("scienceJustification", "")
        n_areas = len(data_field.get("year1Areas", [])) if isinstance(data_field, dict) else ""
        rows.append({
            "survey": s,
            "timestamp": ent.get("timestamp"),
            "filename": ent.get("filename"),
            "n_areas": n_areas,
            "justification_preview": (justification[:120] + "…") if justification and len(justification) > 120 else justification
        })

    # provide expanders to inspect full stored document per submission
    for r in rows:
        with st.expander(f"{r['survey']} — {r.get('timestamp','')}", expanded=False):
            doc = latest_submissions.get(r['survey'])
            st.json(doc)

    allSubmissions = 'LTSYear1_all_surveys'+'_'+str(today.year)+today.strftime('%m')+today.strftime('%d')+'.json'
    
    # make this specific download button appear green like the Save button
    st.markdown(
        """
        <style>
        .green-download-btn .stButton>button[disabled] {
            background-color: #6c757d !important; /* grey */
            color: #ffffff !important;
            font-weight: 600 !important;
            opacity: 0.6 !important;
            border: none !important;
        }
        .green-download-btn .stButton>button:not([disabled]) {
            background-color: #28a745 !important; /* green */
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            padding: 0.7rem 1.2rem !important;
            border-radius: 8px !important;
            box-shadow: 0 6px 18px rgba(40,167,69,0.25) !important;
            border: none !important;
        }
        .green-download-btn .stButton>button:not([disabled]):hover {
            background-color: #218838 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="green-download-btn">', unsafe_allow_html=True)
    st.download_button(
        label="Download all survey submissions",
        data=latest_submissions_json,
        file_name=allSubmissions,
        mime="application/json",
        key="download_all_submissions",
    )
    st.markdown('</div>', unsafe_allow_html=True)
st.divider()
st.header("Example JSON inputs")
st.markdown("""Here are three example polygons you can copy, paste, and edit!

All units are in degrees.
""")

c1, c2, c3 = st.columns((1, 1, 1))
c1.header("Decl. Stripe")
c1.json(      {
        "name": "Demo Stripe",
        "type": "stripe",
        "RA_lower": 0.0,
        "RA_upper": 52.5,
        "Dec_lower":-35.0,
        "Dec_upper":-25.0,
        "t_frac": 0.2,
        "year": 1
      })
c2.header('Single 4MOST Pointing')
c2.json({
    "name": "single_point",
    "type": "point",
    "RA_center":150.125,
    "Dec_center":2.2,
    "t_frac": 0.9,
    "year": 2
})

c3.header("Polygon")
c3.json(      {
        "name": "examplePolygon",
        "type": "polygon",
        "RA": [0.0 ,52.5, 52.5, 0.0],
        "Dec":[-35.0 ,-35.0,-25.0,-25.0],
        "t_frac": 0.2,
        "year": 3
      })