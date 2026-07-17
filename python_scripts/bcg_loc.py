#========================================================================================================================
# Import libraries
import icl_functions

import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

plt.rcParams['figure.dpi'] = 1000
 

#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=5:
    print("Usage: python this_script.py cln coadd bcg_csv output_folder") 
    sys.exit(1)

cln = sys.argv[1]
coadd_path = sys.argv[2]
bcg_coords_csv_name = sys.argv[3]
output_folder = sys.argv[4]

bcg_coords_df = pd.read_csv(bcg_coords_csv_name)

#------------------------------------------------------------------------------------------------------------------------
# NED/SIMBAD query on cluster
#------------------------------------------------------------------------------------------------------------------------

cln_ned_table = Ned.query_object(cln) #Query the cluster name

table_ebv = IrsaDust.get_query_table(cln, section='ebv') #Query galactic extinction info
ebv_mean = table_ebv['ext SFD mean'][0]

# ned_RA = cln_ned_table['RA'][0]       #Get the ned coordinates
# ned_DEC = cln_ned_table['DEC'][0]

ned_z = cln_ned_table['Redshift'][0] #Store the redshift

# For Simbad, 250 arcsec region query (about 951 pix for 0.263 arcsec/px)
simbad = Simbad()
simbad.add_votable_fields("otype","rvz_redshift") #grab object type and redshift(?)
cln_simbad_table = simbad.query_region(cln, radius=250*u.arcsec) 

objtypes = ['BiC','LSB','QSO','rG','ClG','BLL','EmG'] #types to search for

# Sort the possible BCG candidates by order of objtype above
order_map = {t: i for i, t in enumerate(objtypes)}
bcg_simbad_table = cln_simbad_table[np.isin(cln_simbad_table['otype'], objtypes)]
sorted_bcg_simbad_table = bcg_simbad_table[np.argsort([order_map[o] for o in bcg_simbad_table['otype']])]

# Below is for manual source anchoring. As in, searching the name in simbad and pasting here
# mask = [
#     str(x).strip().replace(' ', '') == 'LEDA88678'
#     for x in sorted_bcg_simbad_table['main_id']
# ]

# Get the coords of the first simbad entry
simbad_RA = sorted_bcg_simbad_table['ra'][0]
simbad_DEC = sorted_bcg_simbad_table['dec'][0]

#------------------------------------------------------------------------------------------------------------------------
# Calculating distance
#------------------------------------------------------------------------------------------------------------------------

d_A = cosmo.angular_diameter_distance(ned_z)  # calcuate the angular diameter distance (Mpc) from the cluster redshift

rad_to_arcsec =  206264.806   #arcsec per rad
arcsec_per_pix_decam = 0.263 

# Calculate physical scale: how many kpc per arcsecond?
scale = (d_A.to(u.kpc)*u.radian).value/rad_to_arcsec  # kpc/rad to kpc/arcsecond

size_arcsec = 1.0*arcsec_per_pix_decam  # arcsec/px
size_kpc = size_arcsec*scale  # kpc/px




#------------------------------------------------------------------------------------------------------------------------
# Obtaining/refining BCG coordinates
#------------------------------------------------------------------------------------------------------------------------

print("------------------------------------")
print("Getting BCG coordinates...")

#store the data and header from the input fits 
coadd_data = fits.open(coadd_path)[0].data; coadd_header = fits.open(coadd_path)[0].header

# WCS (World Coordinate System) is same for all griz bands, so just using r 
wcs = WCS(coadd_header)  #get the wcs from the input header  


# Check if cluster is available in BCG coord csv. If not, use SIMBAD to obtain coords.

if cln in bcg_coords_df['Name'].values:

    print(f"{cln} found in the BCG dataframe.")

    bcg_coords = bcg_coords_df[bcg_coords_df['Name'] == cln]

    bcg_RA = bcg_coords['RA'].iloc[0]
    bcg_DEC = bcg_coords['DEC'].iloc[0]

    #Convert RA, DEC coordinates to x, y px coordinates. Floats
    x_bcg, y_bcg = wcs.world_to_pixel_values(bcg_RA, bcg_DEC)

    # Refined coordinates. Searches for brighter pixels within chosen radius. Return integer tuple.
    x_bcg_ref, y_bcg_ref = icl_functions.recenter_bcg_coords(coadd_data, x_bcg, y_bcg, radius=25)


else:

    print(f"{cln} not found in BCG dataframe. Using SIMBAD coordinates.")
    x_bcg, y_bcg = wcs.world_to_pixel_values(simbad_RA, simbad_DEC)

    #Using slightly larger search radius
    x_bcg_ref, y_bcg_ref = icl_functions.recenter_bcg_coords(coadd_data, x_bcg, y_bcg, radius=40)

#calculate a scaled radius to plot cutout of BCG with found coordinates
radius = int(200/size_kpc) #pixels for cutout of image

# Display center/refined center
plt.figure()
plt.imshow(icl_functions.display(coadd_data)*255)
plt.scatter(x_bcg,y_bcg,  marker='o', facecolors='none', edgecolors='#1f77b4', label='Original')
plt.scatter(x_bcg_ref,y_bcg_ref, marker='o', facecolors='none', edgecolors= '#ff7f0e', label='Updated')
plt.gca().invert_yaxis()
plt.xlabel('x (px)')
plt.ylabel('y (px)')
plt.xlim(x_bcg_ref - radius, x_bcg_ref + radius)
plt.ylim(y_bcg_ref - radius, y_bcg_ref + radius)
plt.title(fr'{cln} $r$-band coadd BCG')
plt.legend()
plt.savefig(f'{output_folder}/{cln}_r_coadd_BCG_loc.png', dpi=1000, bbox_inches='tight')
plt.close()


#------------------------------------------------------------------------------------------------------------------------
# Create df of BCG info
#------------------------------------------------------------------------------------------------------------------------

#Store all relevant metrics in a csv, to use in later steps
df = pd.DataFrame({
        'bcg_name': [sorted_bcg_simbad_table['main_id'][0]],
        'bcg_x_pix': [x_bcg_ref],
        'bcg_y_pix': [y_bcg_ref],
        'bcg_RA_deg': [simbad_RA],
        'bcg_DEC_deg': [simbad_DEC],
        'redshift': [ned_z],
        'ext_SFD_mean': [ebv_mean], 
        'arcsec_per_pix': [arcsec_per_pix_decam],
        'kpc_per_arcsec': [scale],
        'kpc_per_pix': [size_kpc]
    })

df.to_csv(f'{output_folder}/{cln}_info.csv', index=False)