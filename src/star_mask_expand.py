#========================================================================================================================
# Import libraries

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 1000
import sys

#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=6:
    print("Usage: python this_script.py cln" \
    " band coadd output_folder lsp_masks_folder") 
    sys.exit(1)

cln = sys.argv[1]
band = sys.argv[2]
coadd_path = sys.argv[3]
output_folder = sys.argv[4]
lsp_masks_folder = sys.argv[5]

#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

mask_path = f'{lsp_masks_folder}/{cln}_{band}55-66_deepCoadd.fits'
mask_data = fits.open(mask_path)[0].data; mask_header = fits.open(mask_path)[0].header


wcs = WCS(mask_header)

# Get size of image
ny, nx = mask_data.shape #pix

ny_arcsec = ny*0.263 #arcsec
nx_arcsec = nx*0.263 #arcsec

ny_deg = ny_arcsec/3600 #deg
nx_deg = nx_arcsec/3600 #deg

# Find center coords
center_x_pix = nx//2
center_y_pix = ny//2

# Convert pix to ra, dec
skycoords = wcs.pixel_to_world(center_x_pix, center_y_pix) 

#Degrees
center_ra_deg = skycoords.ra.to(u.deg)
center_dec_deg = skycoords.dec.to(u.deg)

coord = SkyCoord(ra=center_ra_deg, dec=center_dec_deg, frame='icrs')

# width = u.Quantity(nx_deg, u.deg)
# height = u.Quantity(ny_deg, u.deg)

# r = Gaia.query_object_async(coordinate=coord, width=width, height=height)


# Query  brighter stars to make larger masks for 
query_large_mask = f"""
SELECT source_id, ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
FROM gaiadr3.gaia_source
WHERE
  CONTAINS(
    POINT('ICRS', ra, dec),
    BOX('ICRS', {center_ra_deg.value}, {center_dec_deg.value}, {nx_deg}, {ny_deg})
  ) = 1
AND phot_g_mean_mag <= 13
"""

# Query more stars to mask smaller central regions
query_small_mask = f"""
SELECT source_id, ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
FROM gaiadr3.gaia_source
WHERE
  CONTAINS(
    POINT('ICRS', ra, dec),
    BOX('ICRS', {center_ra_deg.value}, {center_dec_deg.value}, {nx_deg}, {ny_deg})
  ) = 1
AND phot_g_mean_mag <= 15
"""

job_large_mask = Gaia.launch_job(query_large_mask)
job_small_mask = Gaia.launch_job(query_small_mask)

r_large = job_large_mask.get_results()
r_small = job_small_mask.get_results() #This is an astropy table. Preserves units.
print(f"Retrieved {len(r_large)} Gaia sources g <= 13.")
print(f"Retrieved {len(r_small)} Gaia sources g <= 15.")

df_large = r_large.to_pandas() #convert to pandas df. Removes units. 
df_small = r_small.to_pandas()

# Create empty coadd array. Same size
# coadd_star_masks = np.zeros_like(g_mask_data, dtype=int) 


# pixels to mask at each star
radius_large = 150
radius_small = 30



# Empty array to fill with star masks
coadd_star_masks = np.zeros_like(mask_data, dtype=int) 

for star in range(len(df_large)):
    ra_deg = df_large['ra'][star]
    dec_deg = df_large['dec'][star]

    # ra_arcsec = ra_deg*3600
    # dec_arcsec = dec_deg*3600

    x_pix, y_pix = wcs.world_to_pixel_values(ra_deg, dec_deg)

    x_pix = int(np.round(x_pix))
    y_pix = int(np.round(y_pix))

    # y_pix = ra_arcsec/0.263
    # x_pix = dec_arcsec/0.263

    # Define bounding box
    x_min = max(0, x_pix - radius_large)
    x_max = min(nx, x_pix + radius_large + 1)
    y_min = max(0, y_pix - radius_large)
    y_max = min(ny, y_pix + radius_large + 1)

    # Extract sub-image
    sub_img = coadd_star_masks[y_min:y_max, x_min:x_max]

    # Create distance mask
    yy, xx = np.indices(sub_img.shape)
    dx = xx + x_min - x_pix
    dy = yy + y_min - y_pix
    mask = dx**2 + dy**2 <= radius_large**2

    # coadd_star_masks[y_min:y_max, x_min:x_max] |= mask #Create union with the empty mask
    coadd_star_masks[y_min:y_max, x_min:x_max] =  np.logical_or(coadd_star_masks[y_min:y_max, x_min:x_max], mask)
    # g_mask_data = np.logical_or(g_mask_data, coadd_star_masks)


for star in range(len(df_small)):
    ra_deg = df_small['ra'][star]
    dec_deg = df_small['dec'][star]

    # ra_arcsec = ra_deg*3600
    # dec_arcsec = dec_deg*3600

    x_pix, y_pix = wcs.world_to_pixel_values(ra_deg, dec_deg)

    x_pix = int(np.round(x_pix))
    y_pix = int(np.round(y_pix))

    # y_pix = ra_arcsec/0.263
    # x_pix = dec_arcsec/0.263

    # Define bounding box
    x_min = max(0, x_pix - radius_small)
    x_max = min(nx, x_pix + radius_small + 1)
    y_min = max(0, y_pix - radius_small)
    y_max = min(ny, y_pix + radius_small + 1)

    # Extract sub-image
    sub_img = coadd_star_masks[y_min:y_max, x_min:x_max]

    # Create distance mask
    yy, xx = np.indices(sub_img.shape)
    dx = xx + x_min - x_pix
    dy = yy + y_min - y_pix
    mask = dx**2 + dy**2 <= radius_large**2

    # coadd_star_masks[y_min:y_max, x_min:x_max] |= mask #Create union with the empty mask
    coadd_star_masks[y_min:y_max, x_min:x_max] =  np.logical_or(coadd_star_masks[y_min:y_max, x_min:x_max], mask)
    # g_mask_data = np.logical_or(g_mask_data, coadd_star_masks)

updated_mask_data = np.logical_or(mask_data, coadd_star_masks)
updated_mask_data = updated_mask_data.astype(np.int16)

hdu = fits.PrimaryHDU(updated_mask_data, header=mask_header)
hdu.writeto(f'{lsp_masks_folder}/{cln}_{band}55-66_star_updated_lsp_mask.fits', overwrite=True)
