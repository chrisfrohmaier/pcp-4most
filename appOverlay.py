import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from datetime import datetime, date
from astropy.time import Time
from pymongo import MongoClient

st.set_page_config(page_title="PCP Planning Tool", layout="wide")
st.title("PCP Polygons & Planning Tool")

# -------------------------------------------------------------
# Database Helper functions
# -------------------------------------------------------------

def get_all_baseline_versions(mongo_uri, db_name="pcp", coll_name="year1all"):
    client = None
    results = []
    try:
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
    except Exception:
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

# -------------------------------------------------------------
# Process PCP Page
# -------------------------------------------------------------

def render_pcp_processor_page():
    import io
    from pcp_processor import process_pcp_state
    
    st.header("PCP Processor")
    st.markdown("Run the PCP processor pipeline on baseline submissions from the database.")
    
    st.subheader("Processor Parameters")
    col1, col2 = st.columns(2)
    with col1:
        nside_options = [16, 32, 64, 128]
        nside = st.selectbox("Select Map Resolution (NSIDE)", options=nside_options, index=1)
        start_date = st.date_input("Project Start Date", value=date.today())
    with col2:
        plot_proj = st.selectbox("Plot Projection", ["mollweide", "cartesian"])
    
    # Calculate MJD from selected start date
    dt_midnight = datetime.combine(start_date, datetime.min.time())
    start_date_mjd = Time(dt_midnight).mjd
    
    submissions = {}
    try:
        mongo_uri = st.secrets.get("MONGO_URI")
        mongo_db = st.secrets.get("MONGO_DB", "pcp")
        mongo_coll = st.secrets.get("MONGO_COLLECTION", "year1all")
        if mongo_uri:
            submissions = select_baseline_submissions(mongo_uri, mongo_db, mongo_coll, "pcp_processor")
    except Exception:
        pass
        
    if st.button("Run PCP Processor", type="primary"):
        with st.spinner("Processing... This may take a minute"):
            polygon_maps_by_year, res = process_pcp_state(
                nside=nside,
                start_date_mjd=start_date_mjd,
                submissions=submissions,
                plot_proj=plot_proj,
                return_figs=True,
                return_fits=True
            )
            st.session_state["pcp_res"] = res

    # Evaluate results outside the button so they survive Streamlit reruns
    if "pcp_res" in st.session_state:
        res = st.session_state["pcp_res"]
        figs = res.get("figs", {})
            
        if figs.get("polygons") is not None:
            st.subheader("PCP Polygon Overlays")
            st.pyplot(figs["polygons"])
            
        fits_dict = res.get("fits", {})
        if fits_dict:
            from astropy.table import vstack
            st.success("Processing complete!")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            col_dl = st.columns(1)[0]
                
            poly_list = fits_dict.get("poly", [])
            if poly_list:
                combined_poly = vstack(poly_list)
                buf_poly = io.BytesIO()
                combined_poly.write(buf_poly, format='fits')
                col_dl.download_button(
                    label="Download PCP Yearly Weights",
                    data=buf_poly.getvalue(),
                    file_name=f"PCP_all_years_weights_{timestamp}.fits",
                    mime="application/fits"
                )

# -------------------------------------------------------------
# Navigation Router
# -------------------------------------------------------------
page = st.sidebar.radio("Navigation", ["Draw Polygons on 4MOST", "Process PCP"])

if page == "Draw Polygons on 4MOST":
    import handdraw_Polygons
    handdraw_Polygons.render_draw_polygons_page()
    st.stop()
elif page == "Process PCP":
    render_pcp_processor_page()
    st.stop()
