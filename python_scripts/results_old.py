#========================================================================================================================
# Import libraries
import icl_functions
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ManualInterval
from astropy.visualization import AsinhStretch
from astropy.visualization import ZScaleInterval
from astropy.modeling.functional_models import Sersic1D
from astropy.modeling.fitting import LevMarLSQFitter, LMLSQFitter,TRFLSQFitter, DogBoxLSQFitter
from astropy.nddata import block_reduce
from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_Ellipse
from matplotlib.patches import Circle
import numpy as np
import numpy.ma as ma
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from photutils.centroids import (centroid_1dg, centroid_com)
from photutils.aperture import ApertureStats
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry
from photutils.profiles import RadialProfile
from photutils.segmentation import SegmentationImage
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pandas as pd
import pdfplumber
import re
import subprocess 
import sys
import tempfile
import urllib.request

plt.rcParams['figure.dpi'] = 1000


#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=8:
    print("Usage: python this_script.py cln coadd_griz output_folder sextractor_folder") 
    sys.exit(1)

cln = sys.argv[1]
coadd_g_path = sys.argv[2]
coadd_r_path = sys.argv[3]
coadd_i_path = sys.argv[4]
coadd_z_path = sys.argv[5]
# info_csv_path = sys.argv[2]
output_folder = sys.argv[6]
sextractor_folder = sys.argv[7]


#========================================================================================================================
#========================================================================================================================
# FUNCTIONS
#========================================================================================================================
#========================================================================================================================

# This is nearly the same function as in icl_functions. Just different plotting.

def draw_lsb_contours(coadd_data, mask_data, wcs, sb_limit, x_bcg, y_bcg, band, cln, output_folder, scale, unmsk_size = 100, superpix_scale=16, sigma=3, block_method=np.mean, color_im=None):
    """
    Draw magnitude contours on smoothed coadd.
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input array/image. Should be ma.

    sb_limit : float
    band : string
        String label. (e.g. "r", "i")
    cln : str
        Cluster name
    output_folder : str
        output folder path
    superpix_scale : int
        Superpixel binning factor. Default 16 pix.
    sigma : int
        Standard deviation value to use for gaussian kernel smoothing. Default 3 sigma.
    block_method : func
        Method to combine pixels. Default is mean.
    color_im : array
        array holding 3 coadds. For color image figures
        
    Returns:
    --------
    mat_2 : 2D numpy array
        Clipped and stretched image
    """
    arcsec_per_pix_super = superpix_scale*(0.263) #multiply by DECam arcsec per pix

    # Separate masked array
    data = coadd_data
    mask = mask_data.astype(bool)

    h, w = data.shape #Size of coadd 

    # block data. Ignores masked pixels
    data_blocked = block_reduce(np.where(mask, 0.0, data), block_size=superpix_scale, func=np.sum)

    valid_pix_blocked = block_reduce((~mask).astype(float), block_size=superpix_scale, func=np.sum)

    # Mean per superpixel
    data_blocked = data_blocked/valid_pix_blocked
    data_blocked[valid_pix_blocked == 0] = np.nan

    # Mask block if too many pix masked
    mask_blocked = block_reduce(mask.astype(float), block_size=superpix_scale,func=np.mean)
    mask_blocked = mask_blocked > 0.8  

    # smooth data and weights
    data_smoothed = gaussian_filter(np.nan_to_num(data_blocked), sigma=sigma)
    weight_smoothed = gaussian_filter((~mask_blocked).astype(float), sigma=sigma)

    # Renormalize
    data_smoothed /= weight_smoothed
    data_smoothed[weight_smoothed == 0] = np.nan

    data_smoothed_arcsec = data_smoothed/(arcsec_per_pix_super)**2
    data_smoothed_mag = -2.5*np.log10(data_smoothed_arcsec) + 27
    data_smoothed_mag = np.ma.masked_invalid(data_smoothed_mag)

    # convert physical scale radius to pix
    radius = int(unmsk_size/scale)


    # now we need contours

    # modify the wcs to account for resolution drop
    wcs_rebin  = wcs.deepcopy()
    wcs_rebin.wcs.cdelt *= superpix_scale
    wcs_rebin.wcs.crpix /= superpix_scale


    # # Full-res background coadd. Cropped to BCG
    # plt.figure()

    # # contour_color_base ='#ffd166'
    # contour_color_base ="#ffd166"
    # # levels=np.arange(2,9),cmap='#ffd166'
    # levels=np.arange(25,31) #(n,m) -> [n,...,m - 1]
    # base = mcolors.to_rgb(contour_color_base)
    # colors = [tuple(np.clip(c * f, 0, 1) for c in base) for f in np.linspace(1.6, 0.4, len(levels))]
    # plt.imshow(1 - icl_functions.display(data), origin='lower', cmap='gray')
    # cs = plt.contour(data_smoothed_mag, levels=levels, extent=[0, w, 0, h], colors=colors, linewidths=1.2)
    # plt.clabel(cs, fmt='%d mag', fontsize=6)
    # plt.colorbar(label=r'$\mu_0$ (mag/arcsec$^2$)', orientation="vertical") 
    # plt.xlabel('x (pixels)')
    # plt.ylabel('y (pixels)')
    # plt.xlim(x_bcg - radius, x_bcg + radius)
    # plt.ylim(y_bcg - radius, y_bcg + radius)
    # # plt.legend(prop={'size': 12})
    # plt.title(fr"{cln} {band}-Band SB Contours, $\mu_{{{band}}}^{{\rm lim}}$ = {sb_limit:.1f}")
    # plt.tight_layout()
    # plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_fullres_cropped_final.png', dpi=1000, bbox_inches='tight')
    # plt.close()

    if color_im is None:
        # Full-res background coadd. Cropped to BCG
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=wcs)

        contour_color_base ="#ffd166"
        # levels=np.arange(25,31) #(n,m) -> [n,...,m - 1]
        levels=np.arange(25,31.5, 0.5) #(n,m) -> [n,...,m - 1]
        base = mcolors.to_rgb(contour_color_base)
        colors = [tuple(np.clip(c * f, 0, 1) for c in base) for f in np.linspace(1.6, 0.4, len(levels))]
        im = ax.imshow(1 - icl_functions.display(data), origin='lower', cmap='gray')
        cs = ax.contour(data_smoothed_mag, levels=levels, colors=colors, linewidths=1.2, transform = ax.get_transform(wcs_rebin))
        ax.clabel(cs, fmt='%d mag', fontsize=6)
        # cbar = plt.colorbar(im, ax = ax)
        # cbar.set_label(r'$\mu_0$ (mag/arcsec$^2$)') 
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.coords[0].set_format_unit('deg', decimal=True)
        ax.coords[1].set_format_unit('deg', decimal=True)
        ax.coords[0].set_major_formatter('d.dd')
        ax.coords[1].set_major_formatter('d.dd')
        ax.set_xlim(x_bcg - radius, x_bcg + radius)
        ax.set_ylim(y_bcg - radius, y_bcg + radius)
        # plt.xlim(x_bcg - radius, x_bcg + radius)
        # plt.ylim(y_bcg - radius, y_bcg + radius)
        ax.set_title(fr"{cln} {band}-Band SB Contours, $\mu_{{{band}}}^{{\rm lim}}$ = {sb_limit:.1f}")
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_fullres_cropped_final.png', dpi=1000, bbox_inches='tight')
        plt.close()

    else:
        # Full-res background color coadd. Cropped to BCG
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=wcs)

        contour_color_base ="#fffffff2"
        levels=np.arange(25,31.5, 0.5) #(n,m) -> [n,...,m - 1]
        base = mcolors.to_rgb(contour_color_base)
        colors = [tuple(np.clip(c * f, 0, 1) for c in base) for f in np.linspace(1.8, 0.4, len(levels))]
        im = ax.imshow(color_im, origin='lower', cmap='gray')
        cs = ax.contour(data_smoothed_mag, levels=levels, colors=colors, linewidths=1.2, transform = ax.get_transform(wcs_rebin))
        ax.clabel(cs, fmt='%d mag', fontsize=6)
        # cbar = plt.colorbar(im, ax = ax)
        # cbar.set_label(r'$\mu_0$ (mag/arcsec$^2$)') 
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.coords[0].set_format_unit('deg', decimal=True)
        ax.coords[1].set_format_unit('deg', decimal=True)
        ax.coords[0].set_major_formatter('d.dd')
        ax.coords[1].set_major_formatter('d.dd')
        ax.set_xlim(x_bcg - radius, x_bcg + radius)
        ax.set_ylim(y_bcg - radius, y_bcg + radius)
        # plt.xlim(x_bcg - radius, x_bcg + radius)
        # plt.ylim(y_bcg - radius, y_bcg + radius)
        ax.set_title(fr"{cln} {band}-Band SB Contours, $\mu_{{{band}}}^{{\rm \text{{lim}}}}$ = {sb_limit:.1f}")
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_fullres_irg_cropped_final.png', dpi=1000, bbox_inches='tight')
        plt.close()



#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

color_dict = {'g': 'dodgerblue', 'r': 'tomato', 'i': 'purple', 'z': 'goldenrod'} 

info_df = pd.read_csv(f"{output_folder}/{cln}_info.csv")

bands = ['g','r','i','z']

# Load in sb profiles
int_g_df = pd.read_csv(f"{output_folder}/{cln}_g_intensity_profile_final.csv")
int_r_df = pd.read_csv(f"{output_folder}/{cln}_r_intensity_profile_final.csv")
int_i_df = pd.read_csv(f"{output_folder}/{cln}_i_intensity_profile_final.csv")
int_z_df = pd.read_csv(f"{output_folder}/{cln}_z_intensity_profile_final.csv")

int_df = {'g': int_g_df, 'r': int_r_df, 'i': int_i_df, 'z': int_z_df}

# Load in coadds
coadd_g_data = fits.open(coadd_g_path)[0].data; coadd_header = fits.open(coadd_g_path)[0].header
coadd_r_data = fits.open(coadd_r_path)[0].data
coadd_i_data = fits.open(coadd_i_path)[0].data
coadd_z_data = fits.open(coadd_z_path)[0].data

coadd_dict = {'g': coadd_g_data, 'r': coadd_r_data, 'i': coadd_i_data, 'z': coadd_z_data}

# Load in masks
final_iter = 3
mask_g_data = fits.open(f'{output_folder}/g_lsp_sex_mask_bcg_unmask_iter_{final_iter}.fits')[0].data; mask_header = fits.open(f'{output_folder}/g_lsp_sex_mask_bcg_unmask_iter_{final_iter}.fits')[0].header
mask_r_data = fits.open(f'{output_folder}/r_lsp_sex_mask_bcg_unmask_iter_{final_iter}.fits')[0].data
mask_i_data = fits.open(f'{output_folder}/i_lsp_sex_mask_bcg_unmask_iter_{final_iter}.fits')[0].data
mask_z_data = fits.open(f'{output_folder}/z_lsp_sex_mask_bcg_unmask_iter_{final_iter}.fits')[0].data

mask_dict = {'g': mask_g_data, 'r': mask_r_data, 'i': mask_i_data, 'z': mask_z_data}



#------------------------------------------------------------------------------------------------------------------------
# Colors
#------------------------------------------------------------------------------------------------------------------------

print("------------------------------------")
print("Calculating colors...")


# Color calculations
gr_color = int_df['g']['sb_mag_calib'] - int_df['r']['sb_mag_calib']
gi_color = int_df['g']['sb_mag_calib'] - int_df['i']['sb_mag_calib']
ri_color = int_df['r']['sb_mag_calib'] - int_df['i']['sb_mag_calib']
rz_color = int_df['r']['sb_mag_calib'] - int_df['z']['sb_mag_calib']
iz_color = int_df['i']['sb_mag_calib'] - int_df['z']['sb_mag_calib']

gr_color_err = np.sqrt(int_df['g']['sb_mag_calib_err']**2 + int_df['r']['sb_mag_calib_err']**2)
gi_color_err = np.sqrt(int_df['g']['sb_mag_calib_err']**2 + int_df['i']['sb_mag_calib_err']**2)
ri_color_err = np.sqrt(int_df['r']['sb_mag_calib_err']**2 + int_df['i']['sb_mag_calib_err']**2)
rz_color_err = np.sqrt(int_df['r']['sb_mag_calib_err']**2 + int_df['z']['sb_mag_calib_err']**2)
iz_color_err = np.sqrt(int_df['i']['sb_mag_calib_err']**2 + int_df['z']['sb_mag_calib_err']**2)


# Should I bin the colors?


# Plotting colors
fig, ax = plt.subplots(figsize=(8, 6))
# ax.semilogx(sma_vals_arcsec,gr_color, 'go', label = r'$(g - r)_0$')
# ax.semilogx(sma_vals_arcsec,ri_color, 'r+', label = r'$(r - i)_0$')

ax.errorbar(int_df['g']['sma_arcsec'], gr_color, yerr=gr_color_err, color=color_dict['g'], fmt='o', capsize=6, markersize=6, mec='black',label=r'$g - r$')
ax.errorbar(int_df['g']['sma_arcsec'], ri_color, yerr=ri_color_err, color=color_dict['r'], fmt='o', capsize=6, markersize=6, mec='black',label=r'$r - i$')
ax.errorbar(int_df['g']['sma_arcsec'], iz_color, yerr=iz_color_err, color=color_dict['i'], fmt='o', capsize=6, markersize=6, mec='black',label=r'$i - z$')

ax.set_xscale('log')
# ax.set_ylim(-5, 5) 
ax.set_ylim(-1, 2) 

ax.set_xlabel('SMA (arcsec)')
ax.set_ylabel('Color')

# Top x-axis with kpc
ax_top = ax.twiny()
ax_top.set_xscale('log')
ax_top.set_xlim(int_df['g']['sma_kpc'].iloc[0], int_df['g']['sma_kpc'].iloc[-1])
ax_top.set_xlabel('SMA (kpc)')

ax.legend()
ax.set_title(f'{cln} BCG+ICL Color Profile')
plt.tight_layout()
plt.savefig(f'{output_folder}/{cln}_color_profile_final.png', dpi=1000, bbox_inches='tight')
plt.close()


#------------------------------------------------------------------------------------------------------------------------
# 1D Surface Brightness Profiles
#------------------------------------------------------------------------------------------------------------------------


# Plotting surface brightness
fig, ax = plt.subplots(figsize=(8, 6))

for band in bands:
    ax.errorbar(int_df[band]['sma_arcsec'], int_df[band]['sb_mag_calib'], yerr=int_df[band]['sb_mag_calib_err'], color=color_dict[band], 
                fmt='o', capsize=6, markersize=6, mec='black', alpha=0.6, label=fr"${band}$, $\mu_{{lim}}$ = {info_df[f'limiting_mag_{band}'].iloc[0]:.1f}")

ax.set_xscale('log')

ax.set_xlabel('SMA (arcsec)')
ax.set_ylabel(r'$\mu$ (mag/arcsec$^2$)')
ax.set_ylim(19, 31) #Fix y limits

# Top x-axis with kpc
ax_top = ax.twiny()
ax_top.set_xscale('log')
ax_top.set_xlim(int_df['g']['sma_kpc'].iloc[0], int_df['g']['sma_kpc'].iloc[-1])
ax_top.set_xlabel('SMA (kpc)')

ax.invert_yaxis()
ax.tick_params(bottom=True, top=False, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.tick_params(direction="in")
ax.legend(prop={'size': 12})
ax.set_title(f'{cln} BCG+ICL Surface Brightness Profiles', fontsize = 15)
plt.tight_layout()
plt.savefig(f'{output_folder}/{cln}_surface_brightness_1d_final.png', dpi=1000, bbox_inches='tight')
plt.close()

#------------------------------------------------------------------------------------------------------------------------
# SB Contour Maps 
#------------------------------------------------------------------------------------------------------------------------

# Need to take both masked coadd and regular coadd as input

wcs = WCS(coadd_header)

# draw contours over respective bands in grayscale
for band in bands:
    draw_lsb_contours(coadd_data = coadd_dict[band], mask_data = mask_dict[band], wcs = wcs, sb_limit = info_df[f'limiting_mag_{band}'].iloc[0], x_bcg = info_df['bcg_x_pix'][0], y_bcg = info_df['bcg_y_pix'][0], band=band, cln=cln,
                                    output_folder=output_folder, scale = info_df['kpc_per_pix'][0], unmsk_size = 400)
    
coadd_irg_data = np.dstack((
                        # np.flipud(icl_functions.display(coadd_i_data)*255), 
                        # np.flipud(icl_functions.display(coadd_r_data)*255), 
                        # np.flipud(icl_functions.display(coadd_g_data)*255),
                        icl_functions.display(coadd_i_data)*255, 
                        icl_functions.display(coadd_r_data)*255, 
                        icl_functions.display(coadd_g_data)*255,
                    )).astype(np.uint8)

draw_lsb_contours(coadd_dict['r'], mask_dict['r'], wcs = wcs, sb_limit = info_df[f'limiting_mag_r'].iloc[0], x_bcg = info_df['bcg_x_pix'][0], y_bcg = info_df['bcg_y_pix'][0], band='r', cln=cln,
                                    output_folder=output_folder, scale = info_df['kpc_per_pix'][0], unmsk_size = 400, color_im = coadd_irg_data)



#------------------------------------------------------------------------------------------------------------------------
# ICL/BCG Fractions
#------------------------------------------------------------------------------------------------------------------------

print("------------------------------------")
print("Calculating Luminosity Fractions...")


# Converting limiting SB in flux units. counts/px(^2?)
lim_flux_px_dict = {}

for band in bands:
    lim_sb = info_df[f'limiting_mag_{band}'].iloc[0]
    lim_flux_arcsec = 10**((27 - lim_sb)/2.5)
    lim_flux_px = lim_flux_arcsec*info_df['arcsec_per_pix'][0]**2
    lim_flux_px_dict[band] = lim_flux_px


# Compiling profiles and errors into dicts
bcg_icl_prof_dict = {band: int_df[band]['mean_int_back_sub'] for band in bands}
bcg_prof_dict = {band: int_df[band]['mean_int_back_sub_Sersic1D'] for band in bands}
icl_prof_dict = {band: int_df[band]['mean_int_back_sub'] - int_df[band]['mean_int_back_sub_Sersic1D'] for band in bands}

bcg_icl_prof_err_dict = {band: int_df[band]['mean_int_back_sub_err'] for band in bands}
bcg_prof_err_dict = {band: int_df[band]['mean_int_back_sub_err'] for band in bands}
icl_prof_err_dict = {band: int_df[band]['mean_int_back_sub_err'] for band in bands}


# For each band, we need to decide if we're summing the light out to the SB (flux) limit, or 1 Mpc. Whichever is smaller

# First calculate 1 Mpc in px

kpc_to_px = info_df['kpc_per_pix'][0]
mpc_to_px = 1000/kpc_to_px

# And find the closest sma px val that's less than this 

closest_mpc_px_dict = {}

for band in bands:
    if np.any(int_df[band]['sma_pix'] < mpc_to_px):
        closest_mpc_px_dict[band] = int_df[band]['sma_pix'][int_df[band]['sma_pix'] < mpc_to_px].max() #This is the pixel value itself, NOT THE INDEX
    else:
        closest_mpc_px_dict[band] = None  


# Now we take it profile-by-profile to decide the summing limit. 

# Let's first sum all the light in the coadd with only a star mask applied.

# Need the 2D coadd errors
coadd_err_data_dict = {band : fits.open(f'{output_folder}/{band}_err_arr.fits')[0].data for band in bands}

# # Need to subtract the background from these coadds now
# background_dict = {band: np.abs((int_df[band]['mean_int_back_sub'] - int_df[band]['mean_int'])[0]) for band in bands}

# coadd_dict = {band: coadd_dict[band] - background_dict[band] for band in bands}

# And we need to fetch the star masks
starmask_data_dict = {band : fits.open(f'{sextractor_folder}/star_seg_{band}.fits')[0].data for band in bands}
starmask_data_dict = {band: np.where(data != 0, 1, 0) for band, data in starmask_data_dict.items()}

# Total meaning all the light in the cluster, or what I consider it to be...
total_prof_dict = {}
total_prof_err_dict = {}

sma = {band: np.asarray(int_df[band]['sma_pix']) for band in bands}

for band in bands:

    # Need to place a zero px in beginning of sma array for the radial profile x-vals to match size of our arrays
    # radius_mod ends up leading to an issue, so don't use this
    # radius_mod = np.insert(int_df[band]['sma_pix'], 0, 0)

    # Better solution so that radii line up
    # sma = np.asarray(int_df[band]['sma_pix'])
    edges = np.concatenate(([0.], sma[band][:-1] + np.diff(sma[band])/2, [sma[band][-1] + np.diff(sma[band])[-1]/2  if len(sma[band])>1 else sma[band][0]*2 ]))

    # Now create the profile
    rp = RadialProfile(
        data=coadd_dict[band], 
        xycen=(info_df['bcg_x_pix'][0],info_df['bcg_y_pix'][0]),
        radii=edges, 
        error=coadd_err_data_dict[band], 
        mask=starmask_data_dict[band].astype(bool)
        )
    total_prof_dict[band] = rp.profile
    total_prof_err_dict[band] = rp.profile_error

# Need to subtract the background
background_dict = {band: np.abs((int_df[band]['mean_int_back_sub'] - int_df[band]['mean_int'])[0]) for band in bands}

total_prof_dict = {band: total_prof_dict[band] - background_dict[band] for band in bands}
# total_prof_err_dict = {band: total_prof_err_dict[band] - background_dict[band] for band in bands}


# Remove first values for better alignment

total_prof_dict = {band: total_prof_dict[band][1:] for band in bands}
bcg_icl_prof_dict = {band: bcg_icl_prof_dict[band][1:] for band in bands}
bcg_prof_dict = {band: bcg_prof_dict[band][1:] for band in bands}
icl_prof_dict = {band: icl_prof_dict[band][1:] for band in bands}

total_prof_err_dict = {band: total_prof_err_dict[band][1:] for band in bands}
bcg_icl_prof_err_dict = {band: bcg_icl_prof_err_dict[band][1:] for band in bands}
bcg_prof_err_dict = {band: bcg_prof_err_dict[band][1:] for band in bands}
icl_prof_err_dict = {band: icl_prof_err_dict[band][1:] for band in bands}

sma = {band: sma[band][1:] for band in bands}



# Now we integrate to sum the light. For the total cluster light...

total_light_dict = {}
total_light_err_dict = {}

for band in bands:

    # Create a mask for the profile, with True for vals above the limiting flux val
    above = total_prof_dict[band] >= lim_flux_px_dict[band]

    # Now we decide whether to choose the limiting flux limit, or Mpc limit

    # if not np.any(above): #If there are NO values above the threshold
    if np.all(above):  #IF ALL values are above the threshold
        x_cut = sma[band]
        flux_cut = total_prof_dict[band]
        x_mask = np.ones_like(x_cut, dtype=bool)

    # Else, find the first crossing where the profile dips below flux threshold
    else:
        cross_idx = np.argmin(above)

        # x value at first crossing
        x_cross = sma[band][cross_idx]

        # Now choose the minimum val of the two limits
        min_x_lim = min(x_cross, closest_mpc_px_dict[band])
        # print(min_x_lim)

        # Now build mask based on x, NOT flux
        x_mask = sma[band] <= min_x_lim
        # print(x_mask)

        # We then shorten the x,y arrays accordingly
        x_cut = sma[band][x_mask]
        flux_cut = total_prof_dict[band][x_mask]


    # Then we integrate!
    area = simpson(flux_cut, x = x_cut)
    total_light_dict[band] = area

    # Now we'll propagate errors using Monte Carlo sampling
    flux_err_cut = total_prof_err_dict[band][x_mask]


    # Monte Carlo error prop
    n_mc = 500
    areas = []


    for _ in range(n_mc):
        sample_flux = flux_cut + np.random.normal(0, flux_err_cut)
        areas.append(simpson(sample_flux, x=x_cut))

    total_light_err_dict[band] = np.std(areas)


# for band in bands:
#     print(f"{band} cluster flux: {total_light_dict[band]} pm {total_light_err_dict[band]}")





# Now we sum the light from the other profiles.

component_light_list = []
component_light_err_list = []


for prof, prof_err in zip([bcg_icl_prof_dict, bcg_prof_dict, icl_prof_dict],
                            [bcg_icl_prof_err_dict, bcg_prof_err_dict, icl_prof_err_dict]):
    
    light_dict = {}
    light_err_dict = {}

    for band in bands:

        # Create a mask for the profile, with True for vals above the limiting flux val
        above = prof[band] >= lim_flux_px_dict[band]

        # Now we decide whether to choose the limiting flux limit, or Mpc limit

        # if not np.any(above): #If there are NO values above the threshold
        if np.all(above):  #IF ALL values are above the threshold
            x_cut = sma[band]
            flux_cut = prof[band]
            x_mask = np.ones_like(x_cut, dtype=bool)
        # Else, find the first crossing where the profile dips below flux threshold
        else:
            cross_idx = np.argmin(above)

            # x value at first crossing
            x_cross = sma[band][cross_idx]

            # Now choose the minimum val of the two limits
            min_x_lim = min(x_cross, closest_mpc_px_dict[band])

            # Now build mask based on x, NOT flux
            x_mask = sma[band] <= min_x_lim

            # We then shorten the x,y arrays accordingly
            x_cut = sma[band][x_mask]
            flux_cut = prof[band][x_mask]


        # Then we integrate!
        area = simpson(flux_cut, x = x_cut)
        light_dict[band] = area

        # Now we'll propagate errors using Monte Carlo sampling
        flux_err_cut = prof_err[band][x_mask]


        # Monte Carlo error prop
        n_mc = 500
        areas = []


        for _ in range(n_mc):
            sample_flux = flux_cut + np.random.normal(0, flux_err_cut)
            areas.append(simpson(sample_flux, x=x_cut))

        light_err_dict[band] = np.std(areas)
    
    component_light_list.append(light_dict)
    component_light_err_list.append(light_err_dict)

bcg_icl_light_dict = component_light_list[0]; bcg_icl_light_err_dict = component_light_err_list[0] 
bcg_light_dict = component_light_list[1]; bcg_light_err_dict = component_light_err_list[1] 
icl_light_dict = component_light_list[2]; icl_light_err_dict = component_light_err_list[2] 

# Now take the ratios and propagate errors!

bcg_icl_frac_dict = {band: bcg_icl_light_dict[band]/total_light_dict[band] for band in bands}
bcg_frac_dict = {band: bcg_light_dict[band]/total_light_dict[band] for band in bands}
icl_frac_dict = {band: icl_light_dict[band]/total_light_dict[band] for band in bands}

bcg_icl_frac_err_dict = {band: np.sqrt((bcg_icl_frac_dict[band]**2)*
    ((bcg_icl_light_err_dict[band]/bcg_icl_light_dict[band])**2 +
    (total_light_err_dict[band]/total_light_dict[band])**2)) for band in bands}

bcg_frac_err_dict = {band: np.sqrt((bcg_frac_dict[band]**2)*
    ((bcg_light_err_dict[band]/bcg_light_dict[band])**2 +
    (total_light_err_dict[band]/total_light_dict[band])**2)) for band in bands}

icl_frac_err_dict = {band: np.sqrt((icl_frac_dict[band]**2)*
    ((icl_light_err_dict[band]/icl_light_dict[band])**2 +
    (total_light_err_dict[band]/total_light_dict[band])**2)) for band in bands}

for band in bands:
    print(f"{band} BCG+ICL Fraction: {bcg_icl_frac_dict[band]*100:.2f}% pm  {bcg_icl_frac_err_dict[band]*100:.2f}%")
    print(f"{band} BCG Fraction: {bcg_frac_dict[band]*100:.2f}% pm  {bcg_frac_err_dict[band]*100:.2f}%")
    print(f"{band} ICL Fraction: {icl_frac_dict[band]*100:.2f}% pm  {icl_frac_err_dict[band]*100:.2f}%")


# Save into a df/csv

fraction_df = pd.DataFrame({
    "band": bcg_icl_frac_dict.keys(),
    "BCG_ICL_frac": bcg_icl_frac_dict.values(),
    "BCG_frac": bcg_frac_dict.values(),
    "ICL_frac": icl_frac_dict.values(),
    "BCG_ICL_frac_err": bcg_icl_frac_err_dict.values(),
    "BCG_frac_err": bcg_frac_err_dict.values(),
    "ICL_frac_err": icl_frac_err_dict.values()
})

fraction_df.to_csv(f'{output_folder}/{cln}_fractions.csv', index=False)