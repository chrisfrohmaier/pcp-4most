#!/usr/bin/env python3
import argparse
import sys
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u

def main():
    parser = argparse.ArgumentParser(
        description="Find the center and minimum enclosing circle radius (in degrees) for K2 campaign footprints, and output plot/CSV/JSON."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the K2 footprint CSV file (e.g., k2-footprint_mini.csv)"
    )
    parser.add_argument(
        "--plot-out",
        default="k2_campaigns.png",
        help="Output filename for the generated plot (default: k2_campaigns.png)"
    )
    parser.add_argument(
        "--csv-out",
        default="k2_campaign_centers.csv",
        help="Output CSV filename for the results table (default: k2_campaign_centers.csv)"
    )
    parser.add_argument(
        "--json-out",
        default="k2_campaign_centers.json",
        help="Output JSON filename for the PCP/LTS app editor (default: k2_campaign_centers.json)"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Required columns check
    corner_cols = ['ra0', 'dec0', 'ra1', 'dec1', 'ra2', 'dec2', 'ra3', 'dec3']
    required_cols = ['campaign'] + corner_cols
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV.", file=sys.stderr)
            sys.exit(1)

    # Group by campaign
    grouped = df.groupby('campaign')

    results = []
    campaign_data = {}

    for campaign_id, group in sorted(grouped):
        # Extract all RA and Dec coordinates for this campaign
        ras = []
        decs = []
        for i in range(4):
            ras.extend(group[f'ra{i}'].dropna().values)
            decs.extend(group[f'dec{i}'].dropna().values)

        if len(ras) == 0:
            continue

        ras = np.array(ras)
        decs = np.array(decs)

        # 1. Get an initial guess for the center using 3D Cartesian coordinates to avoid RA wrapping issues
        r_ras_init = np.radians(ras)
        r_decs_init = np.radians(decs)

        x = np.cos(r_decs_init) * np.cos(r_ras_init)
        y = np.cos(r_decs_init) * np.sin(r_ras_init)
        z = np.sin(r_decs_init)

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_z = np.mean(z)

        # Normalize back to unit sphere
        norm = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
        if norm == 0:
            ra_guess = 0.0
            dec_guess = 0.0
        else:
            mean_x /= norm
            mean_y /= norm
            mean_z /= norm
            dec_guess = np.degrees(np.arcsin(mean_z))
            ra_guess = np.degrees(np.arctan2(mean_y, mean_x)) % 360.0

        # 2. Shift RAs relative to ra_guess to avoid 0/360 wrapping boundary issue during optimization
        ras_shifted = (ras - ra_guess + 180.0) % 360.0 + ra_guess - 180.0

        # Precompute trigonometric values of the points for efficiency in the optimizer
        r_ras_shifted = np.radians(ras_shifted)
        r_decs = np.radians(decs)
        sin_decs = np.sin(r_decs)
        cos_decs = np.cos(r_decs)

        # 3. Define the objective function to minimize: the maximum spherical angular distance
        def objective(c):
            ra_c, dec_c = c
            r_ra_c = np.radians(ra_c)
            r_dec_c = np.radians(dec_c)
            
            # Spherical law of cosines formula
            cos_sep = np.sin(r_dec_c) * sin_decs + np.cos(r_dec_c) * cos_decs * np.cos(r_ras_shifted - r_ra_c)
            cos_sep = np.clip(cos_sep, -1.0, 1.0)
            sep_rad = np.arccos(cos_sep)
            return np.max(sep_rad)

        # 4. Run Nelder-Mead optimization starting from the Cartesian mean guess
        res = minimize(objective, x0=[ra_guess, dec_guess], method='Nelder-Mead')

        if res.success:
            opt_ra, opt_dec = res.x
            opt_ra = opt_ra % 360.0
            radius_rad = res.fun
            radius_deg = np.degrees(radius_rad)
            
            results.append({
                'campaign': campaign_id,
                'center_ra': opt_ra,
                'center_dec': opt_dec,
                'radius_deg': radius_deg
            })
            # Save for plotting
            campaign_data[campaign_id] = {
                'group': group,
                'center_ra': opt_ra,
                'center_dec': opt_dec,
                'radius_deg': radius_deg
            }
        else:
            print(f"Warning: Optimization failed for Campaign {campaign_id}", file=sys.stderr)

    # Output the results
    print(f"{'Campaign':<10} | {'Center RA (deg)':<15} | {'Center Dec (deg)':<15} | {'Radius (deg)':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['campaign']:<10} | {r['center_ra']:<15.6f} | {r['center_dec']:<15.6f} | {r['radius_deg']:<15.6f}")

    # Save CSV results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.csv_out, index=False)
        print(f"\nResults successfully saved as CSV to {args.csv_out}")

    # Save JSON results for application
    if results:
        json_data = {
            "survey": "S00",
            "scienceJustification": f"Kepler K2 Campaign Footprints generated from {args.csv_file}",
            "author": "Chris + gemini!",
            "year1Areas": [
                {
                    "name": f"K2_{r['campaign']}",
                    "type": "point",
                    "RA_center": float(round(r['center_ra'], 6)),
                    "Dec_center": float(round(r['center_dec'], 6)),
                    "radius": float(round(r['radius_deg'], 2)),
                    "t_frac": 1.0,
                    "year": [1,2]
                }
                for r in results
            ]
        }
        with open(args.json_out, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Results successfully saved as JSON to {args.json_out}")

    # Plotting
    if results:
        print(f"Generating plot and saving to {args.plot_out}...")
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Color cycle for campaigns
        colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
        
        for idx, r in enumerate(results):
            c_id = r['campaign']
            data = campaign_data[c_id]
            color = colors[idx]
            
            # Plot individual CCDs as polygons
            first_ccd = True
            for _, row in data['group'].iterrows():
                corners = []
                for i in range(4):
                    if pd.notna(row[f'ra{i}']) and pd.notna(row[f'dec{i}']):
                        corners.append((row[f'ra{i}'], row[f'dec{i}']))
                
                if len(corners) == 4:
                    # To prevent drawing issues near RA boundary: shift RA relative to the center
                    shifted_corners = []
                    for ra_val, dec_val in corners:
                        ra_s = (ra_val - data['center_ra'] + 180.0) % 360.0 + data['center_ra'] - 180.0
                        shifted_corners.append((ra_s, dec_val))
                    
                    label = f"Campaign {c_id}" if first_ccd else ""
                    poly = Polygon(shifted_corners, closed=True, facecolor=color, edgecolor='black', alpha=0.3, linewidth=0.5, label=label)
                    ax.add_patch(poly)
                    first_ccd = False

            # Generate and plot the enclosing circle on the sphere
            center_coord = SkyCoord(ra=data['center_ra']*u.deg, dec=data['center_dec']*u.deg)
            offset_frame = SkyOffsetFrame(origin=center_coord)
            angles = np.linspace(0, 2*np.pi, 200)
            
            circle_offset = SkyCoord(
                lon=np.cos(angles)*data['radius_deg']*u.deg, 
                lat=np.sin(angles)*data['radius_deg']*u.deg, 
                frame=offset_frame
            )
            circle_coords = circle_offset.transform_to('icrs')
            
            # Shift circle RAs to align with the center
            c_ras = circle_coords.ra.deg
            c_decs = circle_coords.dec.deg
            c_ras_shifted = (c_ras - data['center_ra'] + 180.0) % 360.0 + data['center_ra'] - 180.0
            
            # Plot the enclosing circle
            ax.plot(c_ras_shifted, c_decs, color='red', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4)
            
            # Plot center point
            ax.scatter(data['center_ra'], data['center_dec'], color='red', marker='+', s=80, zorder=5)
            
            # Add label for the campaign
            ax.text(
                data['center_ra'], data['center_dec'] + data['radius_deg'] + 0.15, 
                f"C{c_id}", 
                color='black', fontsize=9, fontweight='bold', ha='center', va='bottom', zorder=6
            )

        # Style the plot
        ax.set_xlabel('RA (deg)', fontsize=12)
        ax.set_ylabel('Dec (deg)', fontsize=12)
        ax.set_title('K2 Campaigns: CCD Footprints, Centers, and Enclosing Field-of-View Circles', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Astronomy convention: RA increases to the left
        ax.invert_xaxis()
        
        # Legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1.0))
        
        plt.tight_layout()
        plt.savefig(args.plot_out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot successfully saved to {args.plot_out}")

if __name__ == "__main__":
    main()
