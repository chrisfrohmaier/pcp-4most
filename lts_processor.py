import json
import argparse
import sys
import os
import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pymongo import MongoClient
import time
from astropy.table import QTable
from astropy import units as u

def lsst_map_to_LTS_format(lsst_map, nside, year_start_mjd, base_mjd):
    """Convert LSST maps to LTS format."""
    valid_mask = ~np.isnan(lsst_map)
    field_id = np.where(valid_mask)[0]
    
    ra, dec = hp.pix2ang(nside, field_id, nest=True, lonlat=True)
    
    # JD corresponding to the start MJD of the year
    jd = np.full(len(field_id), year_start_mjd + 2400000.5)
    LTS_user_weight = lsst_map[valid_mask]
    weight_timescale = np.full(len(field_id), 365.0)
    
    df = pd.DataFrame({
        'field_id': field_id,
        'ra': ra,
        'dec': dec,
        'JD': jd,
        'LTS_user_weight': LTS_user_weight,
        'weight_timescale': weight_timescale
    })
    
    lsst = QTable.from_pandas(df)

    col_units = {
        'field_id': '',
        'ra': 'deg',
        'dec': 'deg',
        'JD': 'd',
        'LTS_user_weight': '',
        'weight_timescale': 'd'
    }

    for k, v in col_units.items():
        lsst[k].unit = u.Unit(v, format='fits')

    lsst.meta = {
        'nside': nside,
        'ordering': 'NESTED',
        'COORDSYS': 'C',
        'JD_zp': base_mjd + 2400000.5,
        'author':'Chris Frohmaier',
        'Date': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    lsst['ra'].info.description = 'The RA of the field in degrees'
    lsst['dec'].info.description = 'The Dec of the field in degrees'
    lsst['JD'].info.description = 'The Julian Date on which the weight will become active'
    lsst['LTS_user_weight'].info.description = (
        'For LSST, this is a binary of 0 or 1. However, this is not a strict zero it is more of a prefer not to go here'
    )
    lsst['weight_timescale'].info.description = 'The timescale (in days) over which the weight is applied.'
    lsst['field_id'].info.description = 'The HEALPIX ID of the field'

    return lsst

def merge_all_LSST_save_to_file(lsst_tables, filename='lts_all_years_weights.fits'):
    """Merges all yearly QTables into a single file and saves it."""
    from astropy.table import vstack
    if not lsst_tables:
        print("No LTS tables to save.")
        return
    combined = vstack(lsst_tables)
    combined.write(filename, overwrite=True)
    print(f"Saved merged LTS weights to {filename}")


def _parse_iso_ts(ts):
    import datetime as _dt
    if isinstance(ts, _dt.datetime):
        return ts
    try:
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def get_latest_submissions_by_survey(mongo_uri, db_name="lts", coll_name="year1all"):
    client = None
    latest = {}
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        coll = client[db_name][coll_name]
        for doc in coll.find({}):
            survey = None
            try:
                survey = doc.get("data", {}).get("survey")
            except Exception:
                survey = None
                
            if not survey: continue
            
            ts = _parse_iso_ts(doc.get("timestamp"))
            if ts is None: continue
            
            cur = latest.get(survey)
            if cur is None or ts > cur["_parsed_ts"]:
                latest[survey] = {"_parsed_ts": ts, "doc": doc}
                
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
        print(f"Error fetching from MongoDB: {e}")
        return {}
    finally:
        try:
            if client: client.close()
        except Exception:
            pass

def get_pixels_for_shape(nside, shape, ra_all, dec_all):
    """Returns pixel indices for a given shape dictionary using geometric bounds."""
    pixels = []
    
    if shape['type'] == 'stripe':
        ra_1 = shape.get('RA_lower', 0)
        ra_2 = shape.get('RA_upper', 0)
        dec_1 = shape.get('Dec_lower', 0)
        dec_2 = shape.get('Dec_upper', 0)
        
        ra_min, ra_max = min(ra_1, ra_2), max(ra_1, ra_2)
        dec_min, dec_max = min(dec_1, dec_2), max(dec_1, dec_2)
        
        # Simple bounding box check on RA and Dec
        # If the stripe appears to logically cross the RA=0 boundary
        if ra_1 > ra_2 and (ra_1 - ra_2) > 180: 
            mask_ra = (ra_all >= ra_1) | (ra_all <= ra_2)
        else:
            mask_ra = (ra_all >= ra_min) & (ra_all <= ra_max)
            
        mask = mask_ra & (dec_all >= dec_min) & (dec_all <= dec_max)
        pixels = np.where(mask)[0]

    elif shape['type'] == 'point':
        ra_center = shape.get('RA_center', 0)
        dec_center = shape.get('Dec_center', 0)
        radius_deg = 2
        
        # Native healpy is perfect for drawing circles on a sphere
        theta = np.deg2rad(90.0 - dec_center)
        phi = np.deg2rad(ra_center)
        vec = hp.ang2vec(theta, phi)
        pixels = hp.query_disc(nside, vec, np.deg2rad(radius_deg), nest=True)

    elif shape['type'] in ('polygon', 'box'):
        ra_arr = shape.get('RA', [])
        dec_arr = shape.get('Dec', [])
        
        if len(ra_arr) >= 3:
            pts = np.column_stack((ra_arr, dec_arr))
            
            # Matplotlib Path is extremely fast, robust, and handles concave shapes 
            # exactly like a web browser UI or shapely does.
            path = Path(pts)
            mask = path.contains_points(np.column_stack((ra_all, dec_all)))
            pixels = np.where(mask)[0]
            
    return pixels

def create_polygon_map(nside, target_year, submissions):
    """Creates a HEALPix map representing the t_frac footprint from MongoDB submissions for a given year."""
    npix = hp.nside2npix(nside)
    # Start with NaNs, meaning no coverage inside this polygon map (distinguishes from t_frac=0.0)
    poly_map = np.full(npix, np.nan, dtype=np.float64)
    
    # Pre-calculate all HEALPix centers for planar point-in-polygon tests
    # Getting RA/Dec for all pixels is very fast
    ra_all, dec_all = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    
    areas_to_process = []
    
    for survey_key, survey_data in submissions.items():
        data = survey_data.get('data', {})
        areas = data.get('year1Areas', [])
        
        for area in areas:
            try:
                area_year = int(area.get('year', 1))
            except Exception:
                area_year = 1
                
            if area_year != target_year:
                continue
                
            t_frac = float(area.get('t_frac', 0.0))
            areas_to_process.append((t_frac, area))
            
    # Sort areas to determine map overriding order.
    # By default, process from smallest to largest t_frac so larger values overwrite smaller ones (taking the max).
    # User Decision: If t_frac is exactly 0.0, it acts as an absolute veto mask and should override everything else.
    # Sorting key: (is_zero, t_frac). This puts non-zeros (False) before zeros (True),
    # ensuring 0.0 is applied last and permanently overwrites any overlapping max footprints.
    areas_to_process.sort(key=lambda x: (x[0] == 0.0, x[0]))
    
    for t_frac, area in areas_to_process:
        # Get pixels inside shape
        pixels = get_pixels_for_shape(nside, area, ra_all, dec_all)
        
        # Assign t_frac to those pixels
        if len(pixels) > 0:
            poly_map[pixels] = t_frac
                
    return poly_map

def convertUserWeightToLTSWeight(userWeight):
    """
    Converts a user-provided weight to an LTS weight.
    The LTS is defined as a weight of 1 == 20% of the avaliable time. This scales such that 
    100% of the time is infinty.

    The function ensures that the input user weight does not exceed 0.95. 
    It then calculates the LTS weight using the formula:
        weightLTS = (userWeight * 4.0) / (1 - userWeight)

    Args:
        userWeight (float): The user-provided weight, expected to be in the range [0, 1].

    Returns:
        float: The calculated LTS weight.

    Notes:
        - If the input user weight is greater than 0.95, it is capped at 0.95 
          to avoid division by zero or excessively large values.
    """
    '''
    This function takes the user weights
    '''
    if userWeight > 0.95:
        userWeight = 0.95
    weightLTS = (userWeight*4.0)/(1-userWeight)
    return weightLTS


def process_app_state(app_state, submissions=None, lts_tfrac=0.5, dec_filter_above=5.0, invert_lsst_Y1_onto_Y2=False, plot_proj="mollweide", return_figs=False, return_fits=False):
    """
    Process the app_state dictionary and optionally operate with MongoDB submissions.
    This function can be called directly when imported into appOverlay.py
    with the app_state dictionary.
    
    Args:
        app_state (dict): The dictionary containing strategy_csv, start_date, 
                          start_date_mjd, nside, and year_thresholds.
        submissions (dict): Optional dictionary of MongoDB submissions for overlays.
        lts_tfrac (float): Value to set for pixels that meet the threshold criteria (default 0.5).
        dec_filter_above (float): Discard observations with Declination strictly greater than this value (default 5.0).
        invert_lsst_Y1_onto_Y2 (bool): Overwrite the Y1 map with the Y2 map (non-zeros set to 0.0) before applying footprints.
        plot_proj (str): Type of projection for diagnostic plots ('mollweide' or 'cartesian').
        return_figs (bool): Return matplotlib figures instead of saving them to disk.
        return_fits (bool): Return astropy table lists instead of saving to FITS.
    """
    if submissions is None:
        submissions = {}
        
    print("--- Processing App State ---")
    print(f"Strategy CSV: {app_state.get('strategy_csv')}")
    print(f"Start Date: {app_state.get('start_date')}")
    print(f"Start Date MJD: {app_state.get('start_date_mjd')}")
    print(f"NSIDE: {app_state.get('nside')}")
    
    year_thresholds = app_state.get('year_thresholds', {})
    if year_thresholds:
        print("Year Thresholds:")
        for year, data in sorted(year_thresholds.items()):
            print(f"  {year}: Threshold = {data.get('threshold')}, Invert Red = {data.get('invert_red')}")
    else:
        print("No year thresholds found.")
        
    print("----------------------------")
    
    lsst_fig = None
    poly_fig = None
    diagnostic_fig = None
    
    # Load the CSV
    strategy_csv = app_state.get('strategy_csv')
    if not strategy_csv:
        print("Error: No strategy CSV provided in app_state.")
        return
        
    csv_path = os.path.join('.', 'strategies', strategy_csv)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return
        
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'fieldDec' in df.columns:
        valid_rows_before = len(df)
        df = df[df['fieldDec'] <= dec_filter_above]
        print(f"Filtered observations with fieldDec > {dec_filter_above}. Remaining rows: {len(df)} (Removed {valid_rows_before - len(df)})")
        
    # Extract RA and Dec
    ra_col = 'fieldRa' if 'fieldRa' in df.columns else 'fieldRA'
    if ra_col not in df.columns or 'fieldDec' not in df.columns:
        print(f"Error: Could not find RA ({ra_col}) or Dec ('fieldDec') columns in the CSV.")
        return
        
    if 'observationStartMJD' not in df.columns:
        print("Error: 'observationStartMJD' column missing from the CSV.")
        return
        
    start_mjd = app_state.get('start_date_mjd')
    if start_mjd is None:
        print("Error: No start_date_mjd found in app_state.")
        return
        
    nside = app_state.get('nside', 16)
    year_thresholds = app_state.get('year_thresholds', {})
    
    hpx_maps_by_year = {}
    print(f"Generating HEALPix maps by year (NSIDE={nside}, Base MJD={start_mjd})...")
    
    # Pass 1: Generate pristine maps according to thresholds and LTS tfrac logic
    year_df_lens = {}
    for year_index in range(5):
        year_num = year_index + 1
        year_key = f"year_{year_num}"
        if year_key not in year_thresholds:
            continue
            
        year_start = start_mjd + year_index * 365
        year_end = start_mjd + (year_index + 1) * 365
        
        # Filter the data for this year
        df_year = df[(df['observationStartMJD'] >= year_start) & (df['observationStartMJD'] < year_end)]
        year_df_lens[year_num] = len(df_year)
        
        ra = df_year[ra_col]
        dec = df_year['fieldDec']
        
        hpx_map = cat2hpx(ra, dec, nside)
        
        # Apply threshold masking
        year_data = year_thresholds[year_key]
        threshold = year_data.get('threshold')
        threshold_type = year_data.get('threshold_type', 'Absolute')
        invert_red = year_data.get('invert_red', False)
        
        if threshold is not None:
            if threshold_type == "Percentage":
                valid_vals = hpx_map[~np.isnan(hpx_map)]
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
                
            if invert_red == False:
                # Valid pixels are > threshold. Mask out everything <= threshold.
                mask = (hpx_map <= absolute_threshold)
            else:
                # Valid pixels are <= threshold. Mask out everything > threshold.
                mask = (hpx_map > absolute_threshold)
                
            hpx_map[mask] = np.nan
            
        # Set all valid pixels (that met the threshold) to the uniform value
        hpx_map[~np.isnan(hpx_map)] = lts_tfrac
            
        hpx_maps_by_year[year_num] = hpx_map
        
    # Apply special rule: Y1 mask adopts Y2 mask footprint (zeroed out)
    if invert_lsst_Y1_onto_Y2 and 1 in hpx_maps_by_year and 2 in hpx_maps_by_year:
        print("Inverting Year 1 footprint: Setting Y1 equal to Y2, with valid values masked to 0.0")
        y2_map = hpx_maps_by_year[2]
        # Valid values (not NaN) become 0.0
        y1_replacement = np.where(~np.isnan(y2_map), 0.0, np.nan)
        hpx_maps_by_year[1] = y1_replacement
        
    # Plot intermediate generated maps (LSST only, before polygons)
    if hpx_maps_by_year:
        lsst_qtable_list = []
        # Give pixels binary values: 0 if 0, 1 if >0. Keep NaNs as NaNs.
        for year, hpx_map in hpx_maps_by_year.items():
            valid_mask = ~np.isnan(hpx_map)
            hpx_map[valid_mask & (hpx_map > 0)] = 1.0
            hpx_map[valid_mask & (hpx_map <= 0)] = 0.0
            
            # Format and collect for LTS before adding polygons
            year_start = start_mjd + (year - 1) * 365
            qtable = lsst_map_to_LTS_format(hpx_map, nside, year_start, start_mjd)
            lsst_qtable_list.append(qtable)

        num_maps = len(hpx_maps_by_year)
        print(f"Plotting {num_maps} intermediate generated maps (LSST only) on a grid...")
        
        rows = 2
        cols = max(1, (num_maps + 1) // rows)
        
        # Create a wider figure to accommodate the columns
        fig1 = plt.figure(figsize=(7 * cols, 5 * rows))
        
        for i, (year, hpx_map) in enumerate(sorted(hpx_maps_by_year.items())):
            kwargs = dict(
                title=f"Year {year} LSST Observations", 
                sub=(rows, cols, i + 1), 
                cmap='viridis', 
                cbar=True, 
                min=0.0,
                max=1.0,
                nest=True,
                rot=[180, 0, 0]
            )
            if plot_proj == "cartesian":
                hp.cartview(hpx_map, **kwargs)
            else:
                hp.mollview(hpx_map, **kwargs)
            
        if return_figs:
            lsst_fig = fig1
        else:
            # Save a quick diagnostic file directly in the current directory
            plot_path = "LSST_year_maps.png"
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Saved intermediate LSST figure to {plot_path}")
            plt.close(fig1)
        
    # Pass 2: Apply polygon masks over the generated maps
    polygon_maps_by_year = {}
    v_func = np.vectorize(convertUserWeightToLTSWeight)
    for year_num, hpx_map in sorted(hpx_maps_by_year.items()):
        # Generate the polygon t_frac map for this year
        poly_map = create_polygon_map(nside, year_num, submissions)
        
        # Apply the user defined LTS scaling function
        in_poly_mask = ~np.isnan(poly_map)
        poly_map[in_poly_mask] = v_func(poly_map[in_poly_mask])
        
        polygon_maps_by_year[year_num] = poly_map
        
        valid_pixels = np.count_nonzero(~np.isnan(hpx_map))
        mapped_poly_pixels = np.count_nonzero(in_poly_mask)
        print(f"  Year {year_num} Map: {year_df_lens[year_num]} obs, {valid_pixels} valid px | Polygon target px: {mapped_poly_pixels}")
    
    # Plot polygon-only maps
    if polygon_maps_by_year:
        num_maps = len(polygon_maps_by_year)
        print(f"Plotting {num_maps} polygon-only maps on a grid...")
        
        rows = 2
        cols = max(1, (num_maps + 1) // rows)
        
        poly_fig = plt.figure(figsize=(7 * cols, 5 * rows))
        
        for i, (year, poly_map) in enumerate(sorted(polygon_maps_by_year.items())):
            kwargs = dict(
                title=f"Year {year} Polygon Overlays", 
                sub=(rows, cols, i + 1), 
                cmap='viridis', 
                cbar=True, 
                min=0.0,
                max=float(np.nanmax(poly_map)),
                nest=True,
                rot=[180, 0, 0]
            )
            if plot_proj == "cartesian":
                hp.cartview(poly_map, **kwargs)
            else:
                hp.mollview(poly_map, **kwargs)
            
        if not return_figs:
            plot_path = "polygon_maps_diagnostic.png"
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Saved polygon diagnostic figure to {plot_path}")
            plt.close(poly_fig)
            poly_fig = None

    # Plot generated maps
    if hpx_maps_by_year:
        num_maps = len(hpx_maps_by_year)
        print(f"Plotting {num_maps} generated maps on a grid...")
        
        rows = 2
        cols = max(1, (num_maps + 1) // rows)
        
        # Create a wider figure to accommodate the columns
        fig2 = plt.figure(figsize=(7 * cols, 5 * rows))
        
        for i, (year, hpx_map) in enumerate(sorted(hpx_maps_by_year.items())):
            kwargs = dict(
                title=f"Year {year} Observations", 
                sub=(rows, cols, i + 1), 
                cmap='viridis', 
                cbar=True, 
                min=0.0,
                max=1.0,
                nest=True,
                rot=[180, 0, 0]
            )
            if plot_proj == "cartesian":
                hp.cartview(hpx_map, **kwargs)
            else:
                hp.mollview(hpx_map, **kwargs)
            
        if return_figs:
            diagnostic_fig = fig2
        else:
            # Save a quick diagnostic file directly in the current directory
            plot_path = "yearly_maps_diagnostic.png"
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Saved diagnostic figure to {plot_path}")
            plt.close(fig2)
            
            # Also attempt to open an interactive plotting window for inspection
            #plt.show()

    # Finally, save the merged LTS generated tables
    poly_qtable_list = []
                
    if polygon_maps_by_year:
        for year, poly_map in sorted(polygon_maps_by_year.items()):
            year_start = start_mjd + (year - 1) * 365
            poly_qtable = lsst_map_to_LTS_format(poly_map, nside, year_start, start_mjd)
            if len(poly_qtable) > 0:
                poly_qtable_list.append(poly_qtable)

    if not return_fits:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if 'lsst_qtable_list' in locals() and lsst_qtable_list:
            merge_all_LSST_save_to_file(lsst_qtable_list, f'lts_all_years_weights_{timestamp}.fits')
        if poly_qtable_list:
            merge_all_LSST_save_to_file(poly_qtable_list, f'User_defined_plan_{timestamp}.fits')
            
    if return_figs or return_fits:
        res = {}
        if return_figs:
            res['figs'] = {'lsst': lsst_fig, 'polygons': poly_fig, 'diagnostic': diagnostic_fig}
        if return_fits:
            res['fits'] = {
                'lsst': lsst_qtable_list if 'lsst_qtable_list' in locals() else [],
                'poly': poly_qtable_list
            }
        return hpx_maps_by_year, res
            
    return hpx_maps_by_year

def cat2hpx(ra, dec, nside):
    npix = hp.nside2npix(nside)
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    
    indices = hp.ang2pix(nside, theta, phi, nest=True)
    idx, counts = np.unique(indices, return_counts=True)

    hpx_map = np.zeros(npix, dtype=np.float64)
    hpx_map[idx] = counts
    hpx_map[hpx_map==0] = np.nan
    return hpx_map

def main():
    parser = argparse.ArgumentParser(description="Process a downloaded app_state JSON file.")
    parser.add_argument(
        "json_file", 
        nargs="?", 
        help="Path to the downloaded appOverlay_state JSON file"
    )
    parser.add_argument(
        "--lts-tfrac",
        type=float,
        default=0.5,
        dest="lts_tfrac",
        help="Value to assign to HEALPix pixels that meet the threshold requirement (default: 0.5)"
    )
    parser.add_argument(
        "--dec-filter-above",
        type=float,
        default=5.0,
        help="Filter out all CSV rows with Declination strictly greater than this value (default: 5.0)"
    )
    parser.add_argument(
        "--invert-lsst-Y1-onto-Y2",
        action="store_true",
        help="Overwrite the Year 1 map footprint to match the Year 2 footprint (setting valid values to 0.0)."
    )
    parser.add_argument(
        "--plot-proj",
        type=str,
        default="mollweide",
        choices=["mollweide", "cartesian"],
        help="Projection type for diagnostic plots (mollweide or cartesian, default: mollweide)"
    )
    
    args = parser.parse_args()
    
    # If run from the command line, load the JSON file and pass the dictionary
    if args.json_file:
        if not os.path.exists(args.json_file):
            print(f"Error: File '{args.json_file}' not found.")
            sys.exit(1)
            
        with open(args.json_file, "r") as f:
            try:
                app_state = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in '{args.json_file}': {e}")
                sys.exit(1)
                
        # Try to fetch Mongo Submissions locally if secrets.toml exists
        submissions = {}
        try:
            import toml
            # Handle possible FileNotFoundError when reading
            if os.path.exists(".streamlit/secrets.toml"):
                secrets = toml.load(".streamlit/secrets.toml")
                mongo_uri = secrets.get("MONGO_URI")
                mongo_db = secrets.get("MONGO_DB", "lts")
                mongo_coll = secrets.get("MONGO_COLLECTION", "year1all")
                if mongo_uri:
                    print(f"Fetching latest MongoDB submissions from local secrets...")
                    submissions = get_latest_submissions_by_survey(mongo_uri, mongo_db, mongo_coll)
                    print(f"Loaded {len(submissions)} latest submissions.")
        except Exception as e:
            print(f"Could not load local MongoDB secrets for independent run: {e}")
            
        # Call the core function with the loaded Python dictionary and submissions
        process_app_state(
            app_state, 
            submissions=submissions, 
            lts_tfrac=args.lts_tfrac, 
            dec_filter_above=args.dec_filter_above,
            invert_lsst_Y1_onto_Y2=args.invert_lsst_Y1_onto_Y2,
            plot_proj=args.plot_proj
        )
    else:
        # If no arguments provided, print help to show how to use it independently
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
