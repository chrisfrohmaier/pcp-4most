import streamlit as st
import pandas as pd
import json
from sqlalchemy import create_engine
import os
import glob
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from astropy.time import Time
from datetime import datetime, time
from plotly.express.colors import sample_colorscale
import shapely.geometry
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon 
from pymongo import MongoClient

st.set_page_config(layout="wide")
st.title("LTS Polygons + LSST Survey Strategies = LTS Plans")

# -------------------------------------------------------------
# exampleApp.py Helper Functions for Plotting Overlays
# -------------------------------------------------------------

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

def rect_corners(xmin, xmax, ymin, ymax, closed=False):
    """
    Return corner coordinates for an axis-aligned rectangle defined by bounds.
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

def _tfrac_to_rgb(t):
    import matplotlib
    t = float(np.clip(t, 0.0, 1.0))
    cmap = matplotlib.colormaps['Spectral']
    rgba = cmap(t)
    rgb = tuple(int(round(255 * v)) for v in rgba[:3])
    return rgb

def _rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def _rgba_str(rgb, alpha=0.25):
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'

def add_polygons_to_fig(fig, data, survey_id, target_year):
    """
    Given a plot object (fig), draws the polygon survey footprints on top of it for a specific year. 
    """
    for i in data.get("year1Areas", []):
        try:
            year_val = int(i.get('year'))
        except Exception:
            # If no year is specified in the DB, default to plotting it in Year 1 so it isn't lost.
            # (or target_year = None means plot regardless of year, which we use for the debug plot below)
            year_val = 1
            
        if target_year is not None and year_val != target_year:
            continue
            
        tfrac = float(i.get('t_frac', 0.0))
        rgb = _tfrac_to_rgb(tfrac)
        hexcol = _rgb_to_hex(rgb)
        fillcol = _rgba_str(rgb, alpha=0.22)
        
        outline_color = "#ffffff"
        outline_width = 3
        inner_width = 2

        if i['type']=='stripe':
            RA_lower = i.get('RA_lower', 0)
            RA_upper = i.get('RA_upper', 0)
            Dec_lower = i.get('Dec_lower', 0)
            Dec_upper = i.get('Dec_upper', 0)
            
            dec_span = abs(Dec_upper - Dec_lower)
            if dec_span < 2.5:
                continue # Skip small stripes
                
            corners = rect_corners(RA_lower, RA_upper, Dec_lower, Dec_upper, closed=True)
            fig.add_trace(go.Scatter(x=corners[:, 0], y=corners[:, 1], showlegend=False, mode="lines",
                                     line=dict(color=outline_color, width=outline_width), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=corners[:, 0], y=corners[:, 1], showlegend=False, mode="lines", fill="toself",
                                     line=dict(color=hexcol, width=inner_width), fillcolor=fillcol,
                                     hoverinfo="name", name=f"{survey_id}<br> t_frac: {tfrac}"))
            
        elif i['type']=='point':
            ra_center = i.get('RA_center', 0)
            dec_center = i.get('Dec_center', 0)
            radius = 1.15
            tissot = plotEllipseTissot(ra_center, dec_center, radius=radius)
            
            fig.add_trace(go.Scatter(x=tissot[:, 0], y=tissot[:, 1], showlegend=False, mode="lines",
                                     line=dict(color=outline_color, width=outline_width), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=tissot[:, 0], y=tissot[:, 1], showlegend=False, mode="lines", fill="toself",
                                     line=dict(color=hexcol, width=inner_width), fillcolor=fillcol,
                                     hoverinfo="name", name=f"{survey_id}<br> t_frac: {tfrac}"))
                                     
        elif i['type']=='polygon' or i['type']=='box':
            RA = i.get('RA', [])
            Dec = i.get('Dec', [])
            if not RA or not Dec:
                continue
            
            # Use raw coordinates directly and ensure the polygon is closed for plotting
            coords = np.array([[r, d] for r, d in zip(RA, Dec)])
            if not np.array_equal(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
            
            fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], showlegend=False, mode="lines",
                                     line=dict(color=outline_color, width=outline_width), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], showlegend=False, mode="lines", fill="toself",
                                     line=dict(color=hexcol, width=inner_width), fillcolor=fillcol,
                                     hoverinfo="name", name=f"{survey_id}<br> t_frac: {tfrac}"))


def _parse_iso_ts(ts):
    import datetime as _dt
    if isinstance(ts, _dt.datetime):
        return ts
    try:
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def get_all_baseline_versions(mongo_uri, db_name="lts", coll_name="year1all"):
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
            if survey and b_val:
                results.append({
                    "survey": survey,
                    "baseline_version": b_val,
                    "timestamp": ts,
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

def select_baseline_submissions(mongo_uri, mongo_db, mongo_coll, key_suffix=""):
    baseline_docs = []
    latest_subs = {}
    if mongo_uri:
        with st.spinner("Fetching DB Baselines..."):
            baseline_docs = get_all_baseline_versions(mongo_uri, mongo_db, mongo_coll)
            
    if baseline_docs:
        def parse_ver(doc):
            try:
                return float(doc["baseline_version"].replace("v", ""))
            except Exception:
                return 0.0
                
        baseline_docs = sorted(baseline_docs, key=lambda x: parse_ver(x), reverse=True)
        opts = {f"[{d['survey']}] {d['baseline_version']} ({str(d.get('timestamp', ''))[:10]})": d for d in baseline_docs}
        
        sel = st.selectbox(
            "Select Baseline Strategy to Include", 
            options=list(opts.keys()), 
            index=0,
            key=f"baseline_selectbox_{key_suffix}"
        )
        if sel:
            d = opts[sel]
            latest_subs[f"{d['survey']}_{d['baseline_version']}"] = {"data": d["data"]}
    else:
        st.warning("No Baseline Submissions loaded or found in DB.")
        
    return latest_subs


def render_lts_processor_page():
    import io
    from lts_processor import process_app_state
    
    st.header("LTS Processor")
    st.markdown("Run the LTS processor pipeline on the configuration set in the Overlays page.")
    
    if "saved_app_state" not in st.session_state:
        st.warning("Please configure the strategy, start date, and thresholds in the Overlays page first.")
        return
        
    app_state = st.session_state["saved_app_state"]
    
    st.subheader("Processor Arguments")
    col1, col2 = st.columns(2)
    with col1:
        invert_Y1 = st.checkbox("Invert LSST Y1 onto Y2")
    with col2:
        dec_filter_above = st.number_input("Declination Filter Above (deg)", value=5.0, step=1.0)
        plot_proj = st.selectbox("Plot Projection", ["mollweide", "cartesian"])
    
    submissions = {}
    try:
        mongo_uri = st.secrets.get("MONGO_URI")
        mongo_db = st.secrets.get("MONGO_DB", "lts")
        mongo_coll = st.secrets.get("MONGO_COLLECTION", "year1all")
        if mongo_uri:
            submissions = select_baseline_submissions(mongo_uri, mongo_db, mongo_coll, "lts_processor")
    except Exception:
        pass
        
    if st.button("Run LTS Processor", type="primary"):
        with st.spinner("Processing... This may take a minute"):
            hpx_maps_by_year, res = process_app_state(
                app_state=app_state,
                submissions=submissions,
                dec_filter_above=dec_filter_above,
                invert_lsst_Y1_onto_Y2=invert_Y1,
                plot_proj=plot_proj,
                return_figs=True,
                return_fits=True
            )
            # Store in session state to persist through download logic re-runs
            st.session_state["lts_res"] = res

    # Evaluate results outside the button so they survive Streamlit reruns
    if "lts_res" in st.session_state:
        res = st.session_state["lts_res"]
        figs = res.get("figs", {})
        
        if figs.get("lsst") is not None:
            st.subheader("LSST Maps")
            st.pyplot(figs["lsst"])
            
        if figs.get("polygons") is not None:
            st.subheader("Polygon Overlays")
            st.pyplot(figs["polygons"])
            
        if figs.get("diagnostic") is not None:
            st.subheader("Final Weight Maps")
            st.pyplot(figs["diagnostic"])
            
        fits_dict = res.get("fits", {})
        if fits_dict:
            from astropy.table import vstack
            from datetime import datetime
            st.success("Processing complete!")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            col_dl1, col_dl2 = st.columns(2)
            
            lsst_list = fits_dict.get("lsst", [])
            if lsst_list:
                combined_lsst = vstack(lsst_list)
                buf_lsst = io.BytesIO()
                combined_lsst.write(buf_lsst, format='fits')
                col_dl1.download_button(
                    label="Download LSST Weights",
                    data=buf_lsst.getvalue(),
                    file_name=f"LSST_all_years_weights_{timestamp}.fits",
                    mime="application/fits"
                )
                
            poly_list = fits_dict.get("poly", [])
            if poly_list:
                combined_poly = vstack(poly_list)
                buf_poly = io.BytesIO()
                combined_poly.write(buf_poly, format='fits')
                col_dl2.download_button(
                    label="Download LTS Yearly Weights",
                    data=buf_poly.getvalue(),
                    file_name=f"User_defined_plan_{timestamp}.fits",
                    mime="application/fits"
                )

# -------------------------------------------------------------
# Navigation
# -------------------------------------------------------------
page = st.sidebar.radio("Navigation", ["Draw Polygons on 4MOST", "Overlay on LSST", "Process LTS"])

if page == "Draw Polygons on 4MOST":
    import handdraw_Polygons
    handdraw_Polygons.render_draw_polygons_page()
    st.stop()
elif page == "Process LTS":
    render_lts_processor_page()
    st.stop()


# -------------------------------------------------------------
# appHealpy.py Logic
# -------------------------------------------------------------

csv_files = glob.glob(os.path.join('.', 'strategies', '*.csv'))
csv_file_names = [os.path.basename(f) for f in sorted(csv_files)]

st.header("Import Configuration")
uploaded_file = st.file_uploader("Upload a saved Configuration JSON to restore state", type=["json"])
if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    if st.session_state.get("last_uploaded_json") != file_content:
        st.session_state["last_uploaded_json"] = file_content
        try:
            loaded_state = json.loads(file_content)
            if "nside" in loaded_state:
                st.session_state["nside_widget"] = loaded_state["nside"]
            if "strategy_csv" in loaded_state:
                if loaded_state["strategy_csv"] in csv_file_names:
                    st.session_state["csv_widget"] = loaded_state["strategy_csv"]
            if "start_date" in loaded_state:
                from datetime import date
                st.session_state["date_widget"] = date.fromisoformat(loaded_state["start_date"])
                
            st.session_state["loaded_thresholds"] = {}
            st.session_state["loaded_threshold_types"] = {}
            yr_thresh = loaded_state.get("year_thresholds", {})
            for y in range(5):
                yr_data = yr_thresh.get(f"year_{y+1}", {})
                if "threshold" in yr_data and yr_data["threshold"] is not None:
                    st.session_state["loaded_thresholds"][y] = yr_data["threshold"]
                if "threshold_type" in yr_data:
                    st.session_state["loaded_threshold_types"][y] = yr_data["threshold_type"]
                if "invert_red" in yr_data:
                    st.session_state[f"invert_yr_{y}"] = yr_data["invert_red"]
            st.session_state["apply_loaded_state"] = True
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")

# DB Connection
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
    latest_submissions = select_baseline_submissions(mongo_uri, mongo_db, mongo_coll, "overlay")
# NSIDE determines the resolution of the HEALPix map
nside_options = [16, 32, 64, 128, 256, 512]
NSIDE = st.select_slider("Select Map Resolution (NSIDE)", options=nside_options, value=32, key="nside_widget")

@st.cache_data
def load_csv(csv_path):
    """Loads data from the specified CSV file."""
    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def cat2hpx_DateDiff(ra, dec, nside):
    npix = hp.nside2npix(nside)
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    
    indices = hp.ang2pix(nside, theta, phi, nest=True)
    idx, counts = np.unique(indices, return_counts=True)

    hpx_map = np.zeros(npix, dtype=np.float64)
    hpx_map[idx] = counts
    hpx_map[hpx_map==0] = np.nan
    return hpx_map

if not csv_files:
    st.error("No CSV files found in ./strategies folder.")
    df = None
else:
    selected_file_name = st.selectbox("Select Strategy CSV", csv_file_names, key="csv_widget")
    selected_csv = os.path.join('.', 'strategies', selected_file_name)
    with st.spinner("Loading CSV..."):
        df = load_csv(selected_csv)

xsize = 800
ysize = xsize // 2
longitude = np.linspace(360,0, xsize)
latitude = np.linspace(-90, 90, ysize)
proj = hp.projector.CartesianProj(xsize=xsize, ysize=ysize)
npix = hp.nside2npix(NSIDE)
vec = hp.pix2vec(NSIDE, np.arange(npix), nest=True)
r = hp.Rotator(rot=[180, 0, 0], deg=True)
vec_rot = r(vec)
new_pix = hp.vec2pix(NSIDE, *vec_rot, nest=True)

if df is not None and not df.empty:
    min_mjd = float(df['observationStartMJD'].min())
    
    # Convert the lowest MJD to a python date to use as the default in the calendar
    try:
        min_date = Time(min_mjd, format='mjd').to_datetime().date()
    except Exception:
        min_date = datetime.today().date()
        
    selected_date = st.date_input("Start Date", value=min_date, key="date_widget")
    dt_midnight = datetime.combine(selected_date, time.min)
    start_date = Time(dt_midnight).mjd
    
    def render_year_overlay(year_index, start_mjd, df_all, ns):
        year_start = start_mjd + year_index * 365
        year_end = start_mjd + (year_index + 1) * 365
        
        st.divider()
        st.subheader(f"Year {year_index+1} Map Overlay")
        
        df_year = df_all[(df_all['observationStartMJD'] >= year_start) & (df_all['observationStartMJD'] < year_end)]
        
        with st.spinner(f"Processing Year {year_index+1}..."):
            ra_col = 'fieldRa' if 'fieldRa' in df_year.columns else 'fieldRA'
            ra = df_year[ra_col]
            dec = df_year['fieldDec']
            
            hp_map_standard = cat2hpx_DateDiff(ra, dec, ns)
            hp_map = hp_map_standard[new_pix]
            vec2pix_func = lambda x, y, z: hp.vec2pix(ns, x, y, z, nest=True)
            image_array = proj.projmap(hp_map, vec2pix_func)
            
            # Slider
            valid_vals = image_array[~np.isnan(image_array)]
            if len(valid_vals) > 0:
                min_val = int(np.floor(np.min(valid_vals)))
                max_val = int(np.ceil(np.max(valid_vals)))
            else:
                min_val, max_val = 0, 1
                
            if min_val == max_val:
                max_val = min_val + 1
                
            thresh_type_key = f"thresh_type_{year_index}"
            
            if st.session_state.get("apply_loaded_state", False):
                loaded_type = st.session_state.get("loaded_threshold_types", {}).get(year_index)
                if loaded_type is not None:
                    st.session_state[thresh_type_key] = loaded_type
            
            if thresh_type_key not in st.session_state:
                st.session_state[thresh_type_key] = "Absolute"
                
            col_t1, col_t2 = st.columns([1, 3])
            with col_t1:
                thresh_type = st.radio(
                    f"Threshold Type (Y{year_index+1})", 
                    ["Absolute", "Percentage"],
                    key=thresh_type_key
                )
            
            if thresh_type == "Absolute":
                slider_min = min_val
                slider_max = max_val
                slider_label = f"Threshold for Red Overlay (Absolute, Y{year_index+1})"
            else:
                slider_min = 0
                slider_max = 100
                slider_label = f"Threshold for Red Overlay (% Discarded, Y{year_index+1})"
                
            if f"slider_yr_{year_index}" not in st.session_state:
                st.session_state[f"slider_yr_{year_index}"] = slider_min if thresh_type == "Absolute" else 0

            if st.session_state.get("apply_loaded_state", False):
                loaded_thr = st.session_state.get("loaded_thresholds", {}).get(year_index)
                if loaded_thr is not None:
                    st.session_state[f"slider_yr_{year_index}"] = max(slider_min, min(slider_max, loaded_thr))
                
            current_val = st.session_state[f"slider_yr_{year_index}"]
            current_val = max(slider_min, min(slider_max, current_val))
            
            with col_t2:
                threshold = st.slider(slider_label, 
                                      min_value=slider_min, 
                                      max_value=slider_max, 
                                      value=current_val,
                                      step=1)
            st.session_state[f"slider_yr_{year_index}"] = threshold
            
            if thresh_type == "Percentage":
                total_counts = np.sum(valid_vals)
                target_discard_sum = total_counts * (threshold / 100.0)
                
                sorted_vals = np.sort(valid_vals)
                cumsum_vals = np.cumsum(sorted_vals)
                
                idx = np.searchsorted(cumsum_vals, target_discard_sum)
                if idx >= len(sorted_vals):
                    idx = len(sorted_vals) - 1
                
                absolute_threshold = sorted_vals[idx] if len(sorted_vals) > 0 else 0
            else:
                absolute_threshold = threshold
            
            invert_red = st.checkbox(f"Invert Red Overlay (Above Threshold) (Year {year_index+1})", 
                                     value=st.session_state.get(f"invert_yr_{year_index}", False))
            st.session_state[f"invert_yr_{year_index}"] = invert_red
            
            fig = go.Figure()
            
            # Base Heatmap
            fig.add_trace(go.Heatmap(
                z=image_array,
                x=longitude,
                y=latitude,
                colorscale='Viridis',
                zmin=800,
                zmax=max_val,
                name="LSST Depth"
            ))
            
            # Highlight Mask (Red)
            if invert_red:
                condition = (image_array > absolute_threshold) & (~np.isnan(image_array))
            else:
                condition = (image_array <= absolute_threshold) & (~np.isnan(image_array))
                
            mask_array = np.where(condition, 1, np.nan)
            fig.add_trace(go.Heatmap(
                z=mask_array,
                x=longitude,
                y=latitude,
                colorscale=[[0, 'rgba(255, 0, 0, 0.5)'], [1, 'rgba(255, 0, 0, 0.5)']],
                showscale=False,
                hoverinfo='skip',
                name=f"Thresholded"
            ))
            
            # Overlays (MongoDB 4MOST Surveys)
            if latest_submissions:
                for survey_key in latest_submissions.keys():
                    dataLatest = latest_submissions[survey_key]['data']
                    survey_name = dataLatest.get('survey', survey_key)
                    # Pass the target year (1-indexed based on year_index)
                    add_polygons_to_fig(fig, dataLatest, survey_name, target_year=year_index+1)
            
            fig.update_layout(
                title=f"Projected NESTED HEALPix Map Year {year_index+1} (NSIDE={ns})",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                xaxis=dict(range=[360, 0]),
                yaxis=dict(range=[-90, 30]),
                showlegend=True,
                uirevision=f"year_{year_index}", # Prevents layout jumps on slider updates
                margin=dict(l=40, r=40, t=50, b=40),
            )

            st.plotly_chart(fig)

    # Sequentially render each of the 5 years
    for y in range(5):
        render_year_overlay(y, start_date, df, NSIDE)
        
    if st.session_state.get("apply_loaded_state", False):
        st.session_state["apply_loaded_state"] = False
        
    st.divider()
    st.header("Export Application State")
    st.markdown("Download the current UI settings and configuration parameters to precisely reproduce this map overlay output.")
    
    # Store settings configuration into a dictionary
    app_state = {
        "strategy_csv": selected_file_name,
        "start_date": selected_date.isoformat(),
        "start_date_mjd": start_date,
        "nside": NSIDE,
        "year_thresholds": {}
    }
    
    # Track the UI values dynamically pulled via their session_state keys in the rendering loop
    for y in range(5):
        slider_val = st.session_state.get(f"slider_yr_{y}", None)
        thresh_type_val = st.session_state.get(f"thresh_type_{y}", "Absolute")
        invert_val = st.session_state.get(f"invert_yr_{y}", False)
        app_state["year_thresholds"][f"year_{y+1}"] = {
            "threshold": slider_val,
            "threshold_type": thresh_type_val,
            "invert_red": invert_val
        }
        
    st.session_state["saved_app_state"] = app_state
    json_state = json.dumps(app_state, indent=4)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="Download Configuration JSON",
        data=json_state,
        file_name=f"appOverlay_state_{timestamp}.json",
        mime="application/json"
    )
        
    # --- Debug: Plot all polygons without background heatmap ---
    st.divider()
    st.subheader("Diagnostic: All Loaded Polygons Only")
    debug_fig = go.Figure()
    
    if latest_submissions:
        for survey_key in latest_submissions.keys():
            dataLatest = latest_submissions[survey_key]['data']
            survey_name = dataLatest.get('survey', survey_key)
            # Pass target_year=None to bypass the year filter check
            add_polygons_to_fig(debug_fig, dataLatest, survey_name, target_year=None)
            
    debug_fig.update_layout(
        title="Diagnostic Polygon Plot",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis=dict(range=[360, 0]),
        yaxis=dict(range=[-90, 30]),
        showlegend=True,
        uirevision="diagnostic",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(debug_fig)
        

