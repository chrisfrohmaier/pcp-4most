import json
import argparse
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from shapely.geometry import Polygon, box

def make_ra_continuous(ra, dec):
    """Adjust RA coordinates to be continuous (removing 360-degree jumps)."""
    unwrapped_ra = np.copy(ra)
    for i in range(1, len(unwrapped_ra)):
        diff = unwrapped_ra[i] - unwrapped_ra[i-1]
        if diff > 180:
            unwrapped_ra[i:] -= 360
        elif diff < -180:
            unwrapped_ra[i:] += 360
    return unwrapped_ra

def convert_galactic_belt(max_dec=5.0):
    print(f"Generating Galactic coordinate belt (clipping Declination <= {max_dec})...")
    # Longitude from -180 to 180 in steps of 5 degrees
    l_step = 5
    l_bottom = np.arange(-180, 180 + l_step, l_step)
    b_bottom = np.full_like(l_bottom, -20)
    
    # Return path (top boundary) from 180 back to -180
    l_top = np.arange(180, -180 - l_step, -l_step)
    b_top = np.full_like(l_top, 20)
    
    l_coords = np.concatenate([l_bottom, l_top])
    b_coords = np.concatenate([b_bottom, b_top])
    
    # Convert to ICRS (RA, Dec)
    galactic_coords = SkyCoord(l=l_coords*u.deg, b=b_coords*u.deg, frame='galactic')
    icrs_coords = galactic_coords.icrs
    ra = icrs_coords.ra.deg
    dec = icrs_coords.dec.deg
    
    # Unwrap RA to make it continuous
    unwrapped_ra = make_ra_continuous(ra, dec)
    
    # Create the Shapely polygon in the unwrapped coordinate space
    poly = Polygon(zip(unwrapped_ra, dec))
    if not poly.is_valid:
        poly = poly.buffer(0)
        
    # We want to split the polygon at RA = 0/360 boundaries and clip Declination.
    # Determine the range of unwrapped RA to know which boxes to intersect with
    min_ra = np.floor(unwrapped_ra.min() / 360.0) * 360.0
    max_ra = np.ceil(unwrapped_ra.max() / 360.0) * 360.0
    
    offsets = np.arange(min_ra, max_ra + 360, 360)
    year1_areas = []
    
    part_counter = 1
    for offset in offsets:
        # Define the boundary box for this segment, clipping Declination <= max_dec
        clip_box = box(offset, -90, offset + 360, max_dec)
        intersection = poly.intersection(clip_box)
        
        if not intersection.is_empty:
            # Handle MultiPolygons if the intersection split it further
            geoms = intersection.geoms if hasattr(intersection, 'geoms') else [intersection]
            for geom in geoms:
                if isinstance(geom, Polygon) and not geom.is_empty:
                    # Extract coordinates and shift RA back to [0, 360]
                    ext_coords = np.array(geom.exterior.coords)
                    ra_vertices = (ext_coords[:, 0] - offset).tolist()
                    dec_vertices = ext_coords[:, 1].tolist()
                    
                    year1_areas.append({
                        "name": f"Galactic_Plane_Part_{part_counter}",
                        "type": "polygon",
                        "RA": [round(r, 4) for r in ra_vertices],
                        "Dec": [round(d, 4) for d in dec_vertices],
                        "t_frac": 1.0,
                        "year": 1
                    })
                    part_counter += 1
                    
    # Construct final PCP-compatible JSON output structure
    pcp_input = {
        "survey": "S00",
        "scienceJustification": f"Galactic plane footprint (l: -180 to 180, b: -20 to 20, Dec <= {max_dec}) converted using Astropy and split at RA boundaries.",
        "year1Areas": year1_areas
    }
    
    output_filename = "galactic_plane_pcp_input.json"
    with open(output_filename, "w") as f:
        json.dump(pcp_input, f, indent=4)
        
    print(f"Successfully created {output_filename} containing {len(year1_areas)} polygon segments.")

def main():
    parser = argparse.ArgumentParser(description="Convert Galactic plane belt to PCP-compatible polygons.")
    parser.add_argument(
        "--max-dec",
        type=float,
        default=5.0,
        help="Maximum Declination limit to clip the polygon (default: 5.0)"
    )
    args = parser.parse_args()
    convert_galactic_belt(max_dec=args.max_dec)

if __name__ == "__main__":
    main()
