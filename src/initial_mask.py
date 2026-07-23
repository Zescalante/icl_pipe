#========================================================================================================================
# Import libraries
import icl_functions

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import sys

plt.rcParams['figure.dpi'] = 1000

#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=9:
    print("Usage: python this_script.py cln band coadd lsp_mask mag_diff info_csv sextractor_folder output_folder") 
    sys.exit(1)

cln = sys.argv[1]
band = sys.argv[2]
coadd_path = sys.argv[3]
lsp_mask_path = sys.argv[4]
mag_diff_path = sys.argv[5]
info_csv_path = sys.argv[6]
sextractor_folder = sys.argv[7]
output_folder = sys.argv[8]

info_df = pd.read_csv(info_csv_path)

#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

#------------------------------------------------------------------------------------------------------------------------
# Fetch coadd, unmask region and run SExtractor
#------------------------------------------------------------------------------------------------------------------------


# Load coadd and its header (WCS)
coadd_data = fits.open(coadd_path)[0].data; coadd_header = fits.open(coadd_path)[0].header


# Load LSP mask and its header
lsp_mask_data = fits.open(lsp_mask_path)[0].data; lsp_mask_header = fits.open(lsp_mask_path)[0].header


# plot the coadd with the complete LSP mask overlaid
plt.figure()
cmap = plt.cm.viridis.copy() 
cmap.set_under(cmap(0))
plt.imshow((icl_functions.display(ma.masked_array(coadd_data, mask=lsp_mask_data)).filled(fill_value=-1))*255, cmap=cmap, vmin=0)
plt.gca().invert_yaxis()
plt.colorbar(label="Intensity", orientation="vertical") 
plt.xlabel('x (px)')
plt.ylabel('y (px)')
plt.title(fr'{cln} LSP-Masked ${band}$-band Coadd')
plt.savefig(f'{output_folder}/{cln}_{band}_coadd_lsp_fully_masked_iter_1.png', dpi=1000, bbox_inches='tight') #hard-coded as first-iteration
plt.close()




# Unmask region around the BCG, scaled per-cluster
# This returns a masked array of partially masked coadd. Do I need this returned?
region_unmsk_data = icl_functions.lsp_region_unmask(coadd_data, lsp_mask_data, info_df['bcg_y_pix'][0], info_df['bcg_x_pix'][0], band=band,
                                                    header=coadd_header, scale = info_df['kpc_per_pix'][0], output_folder=output_folder,
                                                    save_full_mask_im=True, unmsk_size=150, run_num=1)

#run SExtractor on the unmasked regino, and stitch the mask to the LSP mask
updated_mask_data, bcg_mask_val = icl_functions.update_mask(coadd=coadd_data, region_path=f'{output_folder}/coadd_region_cutout_{band}_iter_1.fits', mask=lsp_mask_data,
                                         bcg_y=info_df['bcg_y_pix'][0], bcg_x=info_df['bcg_x_pix'][0], band=band, header=coadd_header,
                                         output_folder=output_folder, sextractor_folder=sextractor_folder, run_num=1, thresh_param='0.5')

# Save the combined LSP/SExtractor mask as fits
update_mask_hdu = fits.PrimaryHDU(updated_mask_data, header=coadd_header)
update_mask_hdu.writeto(f'{output_folder}/{band}_lsp_sex_mask_iter_1.fits', overwrite=True)



# Save the combined lsp/sextractor fully-masked image
plt.figure()
cmap = plt.cm.viridis.copy() 
cmap.set_under(cmap(0))
plt.imshow((icl_functions.display(ma.masked_array(coadd_data, mask=updated_mask_data)).filled(fill_value=-1))*255, cmap=cmap, vmin=0)
plt.gca().invert_yaxis()
plt.colorbar(label="Intensity", orientation="vertical") 
plt.xlabel('x (px)')
plt.ylabel('y (px)')
plt.title(fr'{cln} LSP/SExtractor-Masked ${band}$-band Coadd Iteration 1')
plt.savefig(f'{output_folder}/{cln}_{band}_coadd_lsp_sex_fully_masked_iter_1.png', dpi=1000, bbox_inches='tight')
plt.close()



# Unmask BCG. e.g. set the value of the mask overlaid the BCG to 0. Also convert the entire mask to binary mask
coadd_ma_bcg_unmasked = icl_functions.coadd_unmask_bcg(coadd_data, updated_mask_data, bcg_mask_val, band, header=coadd_header, output_folder=output_folder, save_full_mask_im=True, run_num=1)



# display a cutout with only BCG unmasked
radius = int(300/info_df['kpc_per_pix'][0]) #pixels for cutout of image

#stretch the values for proper visualization
coadd_ma_bcg_unmasked_stretched = icl_functions.display(coadd_ma_bcg_unmasked)
coadd_ma_bcg_unmasked_stretched = coadd_ma_bcg_unmasked_stretched.filled(fill_value=-1)
coadd_ma_bcg_unmasked_stretched = coadd_ma_bcg_unmasked_stretched*255

plt.figure()
cmap = plt.cm.viridis.copy() 
cmap.set_under(cmap(0))
plt.imshow(coadd_ma_bcg_unmasked_stretched, cmap=cmap, vmin=0)
plt.gca().invert_yaxis()
plt.colorbar(label="Intensity", orientation="vertical") 
plt.xlim(info_df['bcg_x_pix'][0] - radius, info_df['bcg_x_pix'][0] + radius)
plt.ylim(info_df['bcg_y_pix'][0] - radius, info_df['bcg_y_pix'][0] + radius)
plt.xlabel('x (px)')
plt.ylabel('y (px)')
plt.title(fr'{cln} Masked ${band}$-band Coadd Iteration 1')
plt.savefig(f'{output_folder}/{cln}_{band}_coadd_bcg_unmasked_iter_1.png', dpi=1000, bbox_inches='tight')
plt.close()



#------------------------------------------------------------------------------------------------------------------------
# Calculate Coadd Input Errors
#------------------------------------------------------------------------------------------------------------------------

#Retrieve the per-pixel errors in magnitude units. We need to convert to flux units.
coadd_input_errors_mag = np.array(icl_functions.get_photometric_errors(file_path = mag_diff_path, band = band)) # Errors in units of mag

print(f'photometric input errors (mag): {coadd_input_errors_mag}')

input_err_fractional_arr = (np.log(10)/(2.5))*coadd_input_errors_mag #sigma flux/flux.


# Generate input error array
input_err_array = np.abs(coadd_data*input_err_fractional_arr[0]) #Units of counts/pix. Same as original coadd.


# Save error array fits fits
err_arr_hdu = fits.PrimaryHDU(input_err_array, header=coadd_header)
err_arr_hdu.writeto(f'{output_folder}/{band}_err_arr.fits', overwrite=True)
