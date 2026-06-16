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
    Map t_frac in [0,1] to an RGB tuple using matplotlib's colormap.
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
        year_raw = i.get('year', 1)
        if isinstance(year_raw, list):
            years = []
            for y in year_raw:
                try:
                    years.append(int(y))
                except (ValueError, TypeError):
                    pass
        else:
            try:
                years = [int(year_raw)]
            except (ValueError, TypeError):
                years = [1]

        # compute color from t_frac (default to 0 if missing)
        tfrac = i.get('t_frac', 0.0)
        rgb = _tfrac_to_rgb(tfrac)
        hexcol = _rgb_to_hex(rgb)
        fillcol = _rgba_str(rgb, alpha=0.22)

        # outline settings (white thin line under the colored line)
        outline_color = "#ffffff"
        outline_width = 3  # slightly larger than inner line so white shows as an outline
        inner_width = 2

        for year_val in years:
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
                target_fig = globals().get('fig', None)

            if target_fig is None:
                continue

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
                        f"Dec span={dec_span:.2f}. Minimum is {min_span}.\n This will not be plotted and is not a valid PCP input."
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
                    name=f"{i.get('name', 'stripe')}<br> t_frac: {tfrac}"
                ))
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
                    name=f"{i.get('name', 'point')}<br> t_frac: {tfrac}"
                ))

            elif i['type']=='polygon' or i['type']=='box':
                RA = list(i['RA'])
                Dec = list(i['Dec'])
                tfrac = i['t_frac']
                if len(RA) > 0 and (RA[0] != RA[-1] or Dec[0] != Dec[-1]):
                    RA.append(RA[0])
                    Dec.append(Dec[0])
                coords = np.column_stack((RA, Dec))

                # white outline trace (no fill)
                target_fig.add_trace(go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    showlegend=False,
                    mode="lines",
                    line=dict(color=outline_color, width=outline_width),
                    hoverinfo='skip',
                ))

                # colored filled trace on top
                target_fig.add_trace(go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    showlegend=False,
                    mode="lines",
                    fill="toself",
                    line=dict(color=hexcol, width=inner_width),
                    fillcolor=fillcol,
                    name=f"{i.get('name', 'polygon')}<br> t_frac: {tfrac}"
                ))
        
        else:
            # print("Please enter a valid shape: 'stripe', 'point', or 'polygon' (or 'box').")
            continue





import pandas as pd
import healpy as hp

# Cache the dataset read so we don't hit the disk constantly
@st.cache_data
def load_qvp_data():
    return pd.read_csv("./visit_plans/visits_SELFIE593.txt", sep=r'\s+', comment='#', 
                       names=['id_tile', 'ra', 'dec', 'pos', 'isky', 'texp', 'texp_ob', 'tob_len', 'irank', 'ntile'])

# Cache the heavy spatial projection
@st.cache_data
def get_grid_map(nside):
    qvp = load_qvp_data()
    hpx_map = cat2hpx(qvp['ra'], qvp['dec'], qvp['texp'], nside)

    proj = hp.projector.CartesianProj(xsize=int(420), ysize=int(210))
    npix = hp.nside2npix(nside)
    vec = hp.pix2vec(nside, np.arange(npix), nest=True)
    r = hp.Rotator(rot=[180, 0, 0], deg=True)
    vec_rot = r(vec)
    new_pix = hp.vec2pix(nside, *vec_rot, nest=True)

    hp_map_rotated = hpx_map[new_pix]
    vec2pix_func = lambda x, y, z: hp.vec2pix(nside, x, y, z, nest=True)
    return proj.projmap(hp_map_rotated, vec2pix_func)

f = open('demoArea.json')
demo_io = f.read()

xsize = 420
ysize = xsize/2
# Reversed to natively match projmap orientation and plot backwards safely
longitude = np.linspace(360, 0, int(xsize)) 
latitude = np.linspace(-90, 90, int(ysize))
mesh = np.meshgrid(longitude, latitude)

def cat2hpx(ra, dec, texp, nside):
    npix = hp.nside2npix(nside)
    indices = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    idx, counts = np.unique(indices, return_counts=True)
    fhSum = np.bincount(indices, weights=texp)
    hpx_map = np.zeros(npix, dtype=np.float64)
    hpx_map[idx] = fhSum[fhSum>0]
    hpx_map[hpx_map==0] = np.nan
    return hpx_map






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

def get_latest_submissions_by_survey(mongo_uri, db_name="pcp", coll_name="year1submissions"):
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

def get_all_baseline_versions(mongo_uri, db_name="pcp", coll_name="year1submissions"):
    """
    Query all Baseline versions across all surveys in the DB.
    """
    client = None
    results = []
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        coll = client[db_name][coll_name]
        for doc in coll.find({"data.baseline_version": {"$exists": True}}):
            survey = doc.get("data", {}).get("survey")
            b_val = doc.get("data", {}).get("baseline_version")
            ts = doc.get("timestamp")
            msg = doc.get("data", {}).get("commit_message", "")
            if survey and b_val:
                results.append({
                    "survey": survey,
                    "baseline_version": b_val,
                    "timestamp": ts,
                    "commit_message": msg,
                    "data": doc.get("data"),
                    "_id": doc.get("_id")
                })
        return results
    except Exception as e:
        return []
    finally:
        try:
            if client: client.close()
        except:
            pass

def render_draw_polygons_page():
    if "editor_key" not in st.session_state:
        st.session_state.editor_key = 0
    if "editor_code" not in st.session_state:
        st.session_state.editor_code = None

    # query MongoDB for latest submissions per survey (if secrets provided)
    try:
        mongo_uri = st.secrets.get("MONGO_URI")
        mongo_db = st.secrets.get("MONGO_DB", "pcp")
        mongo_coll = st.secrets.get("MONGO_COLLECTION", "year1all")
    except Exception:
        mongo_uri = None
        mongo_db = "pcp"
        mongo_coll = "year1all"

    latest_submissions = {}
    if mongo_uri:
        latest_submissions = get_latest_submissions_by_survey(mongo_uri, mongo_db, mongo_coll)

    # convert latest_submissions to a JSON string for download/display (ObjectId/datetime -> str)
    try:
        latest_submissions_json = json.dumps(latest_submissions, indent=4, default=str)
    except Exception:
        # print("Error converting latest submissions to JSON")
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

    if st.session_state.editor_code is None:
        if isinstance(dataDefault, dict):
            st.session_state.editor_code = json.dumps(dataDefault, indent=4)
        else:
            st.session_state.editor_code = str(dataDefault)


    #print(latest_submissions)
    # show a compact summary in the sidebar
    # if latest_submissions:
    #     try:
    #         # display full contents for each survey
    #         st.sidebar.header("Latest submissions by survey")
    #         st.sidebar.json(latest_submissions)
    #     except Exception:
    #         pass

    st.title("PSOC PCP Tool")
    # --- UI Load Baseline ---
    baseline_docs = get_all_baseline_versions(mongo_uri, mongo_db, mongo_coll) if mongo_uri else []
    if baseline_docs:
        baseline_docs = sorted(baseline_docs, key=lambda x: (x["survey"], x["baseline_version"]), reverse=True)
        baseline_opts = {f"[{d['survey']}] {d['baseline_version']} ({str(d.get('timestamp', ''))[:10]})": d for d in baseline_docs}
        
        st.markdown("### Load Existing Baseline")
        colA, colB = st.columns([3, 1])
        with colA:
            selected_baseline_label = st.selectbox("Select Baseline to Load", options=["-- Select --"] + list(baseline_opts.keys()))
        with colB:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Load Baseline into Editor", type="primary"):
                if selected_baseline_label != "-- Select --":
                    sel_doc = baseline_opts[selected_baseline_label]
                    # Format as JSON string and push to session state
                    st.session_state.editor_code = json.dumps(sel_doc["data"], indent=4)
                    st.session_state.editor_key += 1
                    st.rerun()

        if selected_baseline_label != "-- Select --":
            sel_doc = baseline_opts[selected_baseline_label]
            c_msg = sel_doc.get("commit_message", "")
            if c_msg:
                st.info(f"**{sel_doc['baseline_version']} Change Log:** {c_msg}")

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
    response_dict = code_editor(st.session_state.editor_code, lang="json", buttons=custom_btns, height=[10, 20], key=f"editor_{st.session_state.editor_key}")
    #print('Response Dict text', response_dict['text'])

    if response_dict['text'] and response_dict['text'] != st.session_state.editor_code:
        st.session_state.editor_code = response_dict['text']

    # Robust JSON parsing: show parse error to user and fall back to dataDefault
    try:
        raw_text = st.session_state.editor_code
        if not raw_text.strip():
            data = json.loads(str(dataDefault))
        else:
            data = json.loads(raw_text)
        #print('Parsed Data', data)
    except Exception as e:
        st.error(f"Error parsing editor JSON: {e}")
        st.info("Using fallback submission data (demo or latest S00) until JSON is valid.")
        data = dataDefault

    if "last_rendered_code" not in st.session_state:
        st.session_state.last_rendered_code = None
    if "cached_figs" not in st.session_state:
        st.session_state.cached_figs = None
    if "last_nside" not in st.session_state:
        st.session_state.last_nside = None

    # Always fetch the cached map so grid_map_nan is universally available down below
    current_nside = st.session_state.get('draw_nside', 32)
    grid_map_nan = get_grid_map(current_nside)
    zmin = 4
    zmax = np.nanmax(grid_map_nan)

    # Only rebuild the expensive base figures if the JSON code changes or resolution shifts!
    if (st.session_state.editor_code != st.session_state.last_rendered_code or 
        st.session_state.cached_figs is None or
        st.session_state.last_nside != current_nside):
        
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
            title='Year 1 Poor Conditions Program Preference: SELFIE 453',
            clickmode='event+select',
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
            colorscale = 'magma',
            name=""
            ), layout=layout)

        old_title = fig.layout.title.text if (hasattr(fig.layout, "title") and fig.layout.title and getattr(fig.layout.title, "text", None)) else "Year 1 Poor Conditions Program Preference"
        figs = []
        for idx in range(1, 6):
            newf = go.Figure(fig)
            # Apply uirevision to prevent jitter/redrawing on selection updates
            newf.update_layout(title=f"Year {idx}", xaxis_range=[360,0], yaxis_range=[-90,30], uirevision='constant')
            figs.append(newf)

        global fig1, fig2, fig3, fig4, fig5
        fig1, fig2, fig3, fig4, fig5 = figs

        if latest_submissions:
            for i in latest_submissions.keys():
                if i == data.get('survey'):
                    continue # Skip the active survey being edited!
                dataLatest = latest_submissions[i]['data']
                plotPolygons(dataLatest, dataLatest.get('survey', i), allColours=False)
        else:
            st.info("No previous submissions found in the remote DB — only the current edited data will be plotted.")

        plotPolygons(data, data['survey'], allColours=True)

        st.session_state.cached_figs = figs
        st.session_state.last_rendered_code = st.session_state.editor_code
        st.session_state.last_nside = current_nside

    import copy
    # Fast reconstruction of figures for the current run
    active_figs = []
    for f in st.session_state.cached_figs:
        active_figs.append(go.Figure(f))
    fig1, fig2, fig3, fig4, fig5 = active_figs

    st.divider()
    st.header("Step 2: Check output on sky map")
    st.markdown("""
    Inspect the sky map here before moving on to the submission step.

    The goal is to avoid oversubscription in Year 1 any R.A. range, which is indicated by the R.A. Time Pressure plot below the sky map.
    We do not want to spend more than 50% of the available time in any R.A. range.
    R.A. pressure is smoothed over a rolling 30 degrees width.
    """)
    
    st.select_slider("Select Map Resolution (NSIDE)", options=[8, 16, 32, 64, 128], value=32, key="draw_nside")

    ncolors = 256
    cmap = matplotlib.colormaps['Spectral']
    colors = [_rgb_to_hex(tuple(int(round(255 * v)) for v in cmap(i / (ncolors - 1))[:3])) for i in range(ncolors)]
    colorscale = [[i / (ncolors - 1), colors[i]] for i in range(ncolors)]
    z = np.linspace(0.0, 1.0, ncolors).reshape(1, -1)
    tick_vals = [round(i * 0.1, 1) for i in range(11)]
    tick_text = [f"{v:.1f}" for v in tick_vals]

    colorbar_fig = go.Figure(go.Heatmap(
        z=z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            orientation='h',
            thickness=6,              
            len=1,                   
            x=0.5,
            xanchor='center',  
            tickmode='array',
            tickvals=tick_vals,      
            ticktext=tick_text,
            ticks='inside',          
            tickfont=dict(size=10)
        )
    ))
    colorbar_fig.update_layout(
        height=60,
        margin=dict(l=8, r=8, t=28, b=6),
        xaxis=dict(visible=False, range=[-0.5, ncolors - 0.5]),
        yaxis=dict(visible=False),
    )

    st.plotly_chart(colorbar_fig)

    # display the five pre-created figures vertically
    for idx, f in enumerate((fig1, fig2, fig3, fig4, fig5)):
        st.plotly_chart(f, key=f"sky_map_{idx}")

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
    fileOutputName = 'S'+str(surveyNumber)+'_'+'PCPYear1'+'_'+str(today.year)+today.strftime('%m')+today.strftime('%d')+'.json'
    st.write('File name:', fileOutputName)
    json_string = json.dumps(data,indent=4, separators=(',', ': '))

    def save_to_remote_db(json_text, filename):
        """
        Save JSON to a MongoDB collection.
        Expects environment variables (or defaults):
          MONGO_URI          - MongoDB connection string (required)
          MONGO_DB           - database name (default: 'pcp')
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
    st.markdown("### Submit as Official Baseline", unsafe_allow_html=True)
    with st.expander("Freeze Baseline Version"):
        st.write("Submit the current editor JSON as a stable, frozen baseline configuration.")
        
        current_survey = data.get("survey", "S00")
        survey_baselines = [d for d in baseline_docs if d["survey"] == current_survey] if "baseline_docs" in locals() and baseline_docs else []
        
        highest_major = 0
        highest_minor = 0
        if survey_baselines:
            for b in survey_baselines:
                b_str = b["baseline_version"].replace("v", "")
                parts = b_str.split(".")
                if len(parts) >= 2:
                    mj, mn = int(parts[0]), int(parts[1])
                    if mj > highest_major or (mj == highest_major and mn > highest_minor):
                        highest_major = mj
                        highest_minor = mn
        
        if highest_major == 0 and highest_minor == 0:
            next_minor = "v0.1"
            next_major = "v1.0"
        else:
            next_minor = f"v{highest_major}.{highest_minor + 1}"
            next_major = f"v{highest_major + 1}.0"
            
        st.write(f"**Latest recorded baseline for {current_survey}:** {f'v{highest_major}.{highest_minor}' if (highest_major > 0 or highest_minor > 0) else 'None'}")
        bump_type = st.radio("Select Version Increment:", [f"Minor Update ({next_minor})", f"Major Update ({next_major})"], key="bump_type")
        commit_message = st.text_area("Commit Description / Change Log (Optional)", help="Describe what's changed from the previous versions")
        
        if expected_pw is None:
            st.warning("Upload password not configured in secrets. Baseline locking disabled.")
        elif pw != expected_pw:
            st.warning("Please enter the correct password in the section above to unlock Baseline Submission.")
            
        if st.button("Commit Baseline", disabled=save_disabled, type="primary"):
            baseline_tag = next_minor if "Minor Update" in bump_type else next_major
            
            try:
                import copy
                payload = copy.deepcopy(data)
                payload["baseline_version"] = baseline_tag
                if commit_message.strip():
                    payload["commit_message"] = commit_message.strip()
                augmented_json_string = json.dumps(payload, indent=4)
                
                ok, msg = save_to_remote_db(augmented_json_string, fileOutputName)
                if ok:
                    st.success(f"Successfully froze {current_survey} as Baseline {baseline_tag}!")
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Error injecting baseline version: {e}")

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

        allSubmissions = 'PCPYear1_all_surveys'+'_'+str(today.year)+today.strftime('%m')+today.strftime('%d')+'.json'

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