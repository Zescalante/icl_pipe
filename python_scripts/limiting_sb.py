#========================================================================================================================
# Import libraries
import icl_functions

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.stats import norm
plt.rcParams['figure.dpi'] = 1000

import pandas as pd
import sys

#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!= 8:
    print("Usage: python this_script.py cln band coadd mask err_arr info_csv output_folder") 
    sys.exit(1)

cln = sys.argv[1]
band = sys.argv[2]
coadd_path = sys.argv[3]
mask_path = sys.argv[4]
err_arr_path = sys.argv[5]
info_csv_path = sys.argv[6]
output_folder = sys.argv[7]

info_df = pd.read_csv(info_csv_path)

#========================================================================================================================
#========================================================================================================================
# FUNCTIONS
#========================================================================================================================
#========================================================================================================================

#Method from Román, Trujillo, & Montes (2020)
def lim_mag(sigma,  pix_scale, sigma_mult = 3, pix_scale_box_arcsec = 10):
    
    return -2.5*np.log10(sigma_mult*sigma/(pix_scale*pix_scale_box_arcsec)) + 27


#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

color_dict = {'g': 'dodgerblue', 'r': 'tomato', 'i': 'purple', 'z': 'goldenrod'}

# Load in coadd, mask, errors

coadd_data = fits.open(coadd_path)[0].data; coadd_header = fits.open(coadd_path)[0].header
mask_data = fits.open(mask_path)[0].data
err_data = fits.open(err_arr_path)[0].data


# Procedure to mask a large region centered on BCG. Basically the exact same as icl_functions.lsp_region_unmask()

# convert physical scale radius to pix
radius = int(1000/info_df['kpc_per_pix'][0])

# Get size of image
ny, nx = coadd_data.shape

x_bcg = info_df['bcg_x_pix'][0]
y_bcg = info_df['bcg_y_pix'][0]

# Define bounding box
x_min = max(0, x_bcg - radius)
x_max = min(nx, x_bcg + radius + 1)
y_min = max(0, y_bcg - radius)
y_max = min(ny, y_bcg + radius + 1)

# Extract sub-image
sub_img = coadd_data[y_min:y_max, x_min:x_max]

# Create distance mask
yy, xx = np.indices(sub_img.shape)
dx = xx + x_min - x_bcg
dy = yy + y_min - y_bcg
mask = dx**2 + dy**2 <= radius**2

region_mask_fullsize = np.zeros_like(mask_data, dtype=bool)
# region_mask_fullsize[y_min:y_max, x_min:x_max] = mask #if you want a circular region mask
region_mask_fullsize[y_min:y_max, x_min:x_max] = True #if you want a square region mask


# Mask region centered on BCG
# mask_arr_center_unmsk = np.where(mask, 0, mask_arr)
mask_arr_center_msk = mask_data.copy()
mask_arr_center_msk[region_mask_fullsize] = 1


plt.figure()
cmap = plt.cm.viridis.copy() 
cmap.set_under(cmap(0))
plt.imshow((icl_functions.display(ma.masked_array(coadd_data, mask=mask_arr_center_msk)).filled(fill_value=-1))*255, cmap=cmap)
plt.gca().invert_yaxis()
plt.colorbar(label="Intensity", orientation="vertical") 
# plt.xlim(x_bcg_ref - 500, x_bcg_ref + 500)
# plt.ylim(y_bcg_ref - 500, y_bcg_ref + 500)
plt.xlabel('x (px)')
plt.ylabel('y (px)')
plt.title(fr'Center-Masked ${band}$-band Coadd')
plt.savefig(f'{output_folder}/{band}_coadd_center_patch_masked.png', dpi=1000, bbox_inches='tight')
plt.close()


center_masked_ma = ma.masked_array(coadd_data, mask=mask_arr_center_msk).filled(np.nan)
center_masked_err_ma = ma.masked_array(err_data, mask=mask_arr_center_msk).filled(np.nan)

# Save the center-masked ma as fits, just in case
hdu = fits.PrimaryHDU(center_masked_ma, header=coadd_header)
hdu.writeto(f'{output_folder}/{band}_center_patch_masked.fits', overwrite=True)

#flattening array to 1D
coadd_data_1d = center_masked_ma.flatten()
coadd_data_1d = coadd_data_1d[~np.isnan(coadd_data_1d)]

err_data_1d = center_masked_err_ma.flatten()
err_data_1d = err_data_1d[~np.isnan(err_data_1d)]


# Original method
# mu, std = norm.fit(coadd_data_1d) 


# Aggressive clipping to get sky-dominated std
mu, median_sky, std = sigma_clipped_stats(
    coadd_data_1d,
    sigma=3.0,          # deviations for clipping
    maxiters=10,        # number of sigma clips to perform
    cenfunc='median',   # computes center value for clipping
    stdfunc='std'       # compute std about center value
)

#calculate limiing SB
limiting_sb = lim_mag(std, info_df['arcsec_per_pix'][0])



#plotting histogram with pdf
x = np.linspace(-30, 30, 3500)
pdf = norm.pdf(x, mu, std)

bin_num = 4000

plt.hist(coadd_data_1d, bins=bin_num, density=True, color=color_dict[band], edgecolor=color_dict[band], histtype='step', label=fr'DECam ${band}$; Limiting SB: {limiting_sb:.2f} mag')
plt.plot(x, pdf, color=color_dict[band], linestyle = '--', linewidth=2, label=f'Fit: $\\mu={mu:.3f}$, $\\sigma={std:.3f}$')

plt.axvline(x=0, color='k', linestyle='--', linewidth=1) #horizontal line marking over/undersubtraction
plt.legend(fontsize=8)
# plt.xlim(-1,1)
plt.xlim(-5*std, 5*std)
plt.title("Histogram of Intensities")
plt.xlabel("Counts")
plt.ylabel("Density")
# plt.show()
plt.savefig(f'{output_folder}/{band}_masked_coadd_background_hist.png', dpi=1000, bbox_inches='tight')
plt.close()

#Export the derived sb limiting value to the info df
info_df[f'limiting_mag_{band}'] = limiting_sb
info_df.to_csv(f'{output_folder}/{cln}_info.csv', index=False)