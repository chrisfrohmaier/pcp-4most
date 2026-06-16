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

def polygon_map_to_PCP_format(pcp_map, nside, year_start_mjd, base_mjd):
    """Convert HEALPix PCP maps to PCP FITS table format."""
    valid_mask = ~np.isnan(pcp_map)
    field_id = np.where(valid_mask)[0]
    
    ra, dec = hp.pix2ang(nside, field_id, nest=True, lonlat=True)
    
    # JD corresponding to the start MJD of the year
    jd = np.full(len(field_id), year_start_mjd + 2400000.5)
    PCP_user_weight = pcp_map[valid_mask]
    weight_timescale = np.full(len(field_id), 365.0)
    
    df = pd.DataFrame({
        'field_id': field_id,
        'ra': ra,
        'dec': dec,
        'JD': jd,
        'PCP_user_weight': PCP_user_weight,
        'weight_timescale': weight_timescale
    })
    
    table = QTable.from_pandas(df)

    col_units = {
        'field_id': '',
        'ra': 'deg',
        'dec': 'deg',
        'JD': 'd',
        'PCP_user_weight': '',
        'weight_timescale': 'd'
    }

    for k, v in col_units.items():
        table[k].unit = u.Unit(v, format='fits')

    table.meta = {
        'nside': nside,
        'ordering': 'NESTED',
        'COORDSYS': 'C',
        'JD_zp': base_mjd + 2400000.5,
        'author': 'Chris Frohmaier',
        'Date': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    table['ra'].info.description = 'The RA of the field in degrees'
    table['dec'].info.description = 'The Dec of the field in degrees'
    table['JD'].info.description = 'The Julian Date on which the weight will become active'
    table['PCP_user_weight'].info.description = 'The PCP weight converted from the user-defined t_frac.'
    table['weight_timescale'].info.description = 'The timescale (in days) over which the weight is applied.'
    table['field_id'].info.description = 'The HEALPIX ID of the field'

    return table

def merge_all_PCP_save_to_file(pcp_tables, filename='pcp_all_years_weights.fits'):
    """Merges all yearly QTables into a single file and saves it."""
    from astropy.table import vstack
    if not pcp_tables:
        print("No PCP tables to save.")
        return
    combined = vstack(pcp_tables)
    combined.write(filename, overwrite=True)
    print(f"Saved merged PCP weights to {filename}")


def _parse_iso_ts(ts):
    import datetime as _dt
    if isinstance(ts, _dt.datetime):
        return ts
    try:
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def get_latest_submissions_by_survey(mongo_uri, db_name="pcp", coll_name="year1all"):
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
        
        if ra_1 > ra_2 and (ra_1 - ra_2) > 180: 
            mask_ra = (ra_all >= ra_1) | (ra_all <= ra_2)
        else:
            mask_ra = (ra_all >= ra_min) & (ra_all <= ra_max)
            
        mask = mask_ra & (dec_all >= dec_min) & (dec_all <= dec_max)
        pixels = np.where(mask)[0]

    elif shape['type'] in ('point', 'circle'):
        ra_center = shape.get('RA_center', 0)
        dec_center = shape.get('Dec_center', 0)
        radius_deg = shape.get('radius', 1.15)
        
        theta = np.deg2rad(90.0 - dec_center)
        phi = np.deg2rad(ra_center)
        vec = hp.ang2vec(theta, phi)
        pixels = hp.query_disc(nside, vec, np.deg2rad(radius_deg), nest=True)

    elif shape['type'] in ('polygon', 'box'):
        ra_arr = shape.get('RA', [])
        dec_arr = shape.get('Dec', [])
        
        if len(ra_arr) >= 3:
            pts = np.column_stack((ra_arr, dec_arr))
            path = Path(pts)
            mask = path.contains_points(np.column_stack((ra_all, dec_all)))
            pixels = np.where(mask)[0]
            
    return pixels

def create_polygon_map(nside, target_year, submissions, default_to_zero=False):
    """Creates a HEALPix map representing the t_frac footprint from MongoDB submissions for a given year."""
    npix = hp.nside2npix(nside)
    if default_to_zero:
        poly_map = np.zeros(npix, dtype=np.float64)
    else:
        poly_map = np.full(npix, np.nan, dtype=np.float64)
    
    ra_all, dec_all = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    
    areas_to_process = []
    
    for survey_key, survey_data in submissions.items():
        data = survey_data.get('data', {})
        areas = data.get('year1Areas', [])
        
        for area in areas:
            year_raw = area.get('year', 1)
            if isinstance(year_raw, list):
                years_list = []
                for y in year_raw:
                    try:
                        years_list.append(int(y))
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    years_list = [int(year_raw)]
                except (ValueError, TypeError):
                    years_list = [1]
                    
            if target_year not in years_list:
                continue
                
            t_frac = float(area.get('t_frac', 0.0))
            areas_to_process.append((t_frac, area))
            
    areas_to_process.sort(key=lambda x: (x[0] == 0.0, x[0]))
    
    for t_frac, area in areas_to_process:
        pixels = get_pixels_for_shape(nside, area, ra_all, dec_all)
        if len(pixels) > 0:
            poly_map[pixels] = t_frac
                
    return poly_map

def convertUserWeightToPCPWeight(userWeight):
    """Converts a user-provided weight to a PCP weight."""
    if userWeight > 0.95:
        userWeight = 0.95
    weightPCP = (userWeight*4.0)/(1-userWeight)
    return weightPCP

def process_pcp_state(nside, start_date_mjd, submissions=None, plot_proj="mollweide", default_to_zero=False, return_figs=False, return_fits=False):
    """
    Process the polygon configurations from submissions and generate PCP FITS files.
    """
    if submissions is None:
        submissions = {}
        
    print("--- Processing PCP State ---")
    print(f"Start Date MJD: {start_date_mjd}")
    print(f"NSIDE: {nside}")
    print(f"Default to Zero: {default_to_zero}")
    print("----------------------------")
    
    poly_fig = None
    polygon_maps_by_year = {}
    v_func = np.vectorize(convertUserWeightToPCPWeight, otypes=[float])
    
    for year_num in range(1, 6):
        # Generate the polygon t_frac map for this year
        poly_map = create_polygon_map(nside, year_num, submissions, default_to_zero=default_to_zero)
        
        # Apply the user defined PCP scaling function
        in_poly_mask = ~np.isnan(poly_map)
        if np.any(in_poly_mask):
            poly_map[in_poly_mask] = v_func(poly_map[in_poly_mask])
        
        polygon_maps_by_year[year_num] = poly_map
        mapped_poly_pixels = np.count_nonzero(in_poly_mask)
        print(f"  Year {year_num} Map: Polygon target px: {mapped_poly_pixels}")
    
    # Plot polygon-only maps
    if polygon_maps_by_year:
        num_maps = len(polygon_maps_by_year)
        rows = 2
        cols = max(1, (num_maps + 1) // rows)
        
        poly_fig = plt.figure(figsize=(7 * cols, 5 * rows))
        
        for i, (year, poly_map) in enumerate(sorted(polygon_maps_by_year.items())):
            poly_max = float(np.nanmax(poly_map)) if np.any(~np.isnan(poly_map)) else 1.0
            kwargs = dict(
                title=f"Year {year} Polygon Overlays", 
                sub=(rows, cols, i + 1), 
                cmap='viridis', 
                cbar=True, 
                min=0.0,
                max=poly_max,
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

    # Save the merged PCP weights tables
    poly_qtable_list = []
                
    if polygon_maps_by_year:
        for year, poly_map in sorted(polygon_maps_by_year.items()):
            year_start = start_date_mjd + (year - 1) * 365
            poly_qtable = polygon_map_to_PCP_format(poly_map, nside, year_start, start_date_mjd)
            if len(poly_qtable) > 0:
                poly_qtable_list.append(poly_qtable)

    if not return_fits:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if poly_qtable_list:
            merge_all_PCP_save_to_file(poly_qtable_list, f'pcp_all_years_weights_{timestamp}.fits')
            
    if return_figs or return_fits:
        res = {}
        if return_figs:
            res['figs'] = {'polygons': poly_fig}
        if return_fits:
            res['fits'] = {
                'poly': poly_qtable_list
            }
        return polygon_maps_by_year, res
            
    return polygon_maps_by_year

def main():
    parser = argparse.ArgumentParser(description="Process a PCP configuration.")
    parser.add_argument(
        "--nside",
        type=int,
        default=32,
        help="HEALPix NSIDE resolution (default: 32)"
    )
    parser.add_argument(
        "--start-date-mjd",
        type=float,
        default=60000.0,
        dest="start_date_mjd",
        help="Project start date in MJD (default: 60000.0)"
    )
    parser.add_argument(
        "--plot-proj",
        type=str,
        default="mollweide",
        choices=["mollweide", "cartesian"],
        help="Projection type for diagnostic plots (mollweide or cartesian, default: mollweide)"
    )
    parser.add_argument(
        "--default-to-zero",
        action="store_true",
        dest="default_to_zero",
        help="Set all HEALPix pixels to 0 by default, instead of NaN"
    )
    
    args = parser.parse_args()
    
    # Try to fetch Mongo Submissions locally if secrets.toml exists
    submissions = {}
    try:
        import toml
        if os.path.exists(".streamlit/secrets.toml"):
            secrets = toml.load(".streamlit/secrets.toml")
            mongo_uri = secrets.get("MONGO_URI")
            mongo_db = secrets.get("MONGO_DB", "pcp")
            mongo_coll = secrets.get("MONGO_COLLECTION", "year1all")
            if mongo_uri:
                print(f"Fetching latest MongoDB submissions from local secrets...")
                submissions = get_latest_submissions_by_survey(mongo_uri, mongo_db, mongo_coll)
                print(f"Loaded {len(submissions)} latest submissions.")
    except Exception as e:
        print(f"Could not load local MongoDB secrets: {e}")
        
    process_pcp_state(
        nside=args.nside,
        start_date_mjd=args.start_date_mjd,
        submissions=submissions,
        plot_proj=args.plot_proj,
        default_to_zero=args.default_to_zero
    )

if __name__ == "__main__":
    main()
