#========================================================================================================================
# Import libraries
import icl_functions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import block_reduce
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter
import pandas as pd
import sys


plt.rcParams['figure.dpi'] = 500

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


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
        The input array/image. 
    mask_data : 2D numpy array
        The input mask data.
    wcs : object
        WCS header object.
    sb_limit : float
    x_bcg, y_bcg : floats
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
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection=wcs)

        contour_color_base ="#ffd166"
        # levels=np.arange(25,31) #(n,m) -> [n,...,m - 1]
        levels=np.arange(25,31.5, 0.5) #(n,m) -> [n,...,m - 1]
        base = mcolors.to_rgb(contour_color_base)
        colors = [tuple(np.clip(c * f, 0, 1) for c in base) for f in np.linspace(1.6, 0.4, len(levels))]
        im = ax.imshow(1 - icl_functions.display(data), origin='lower', cmap='gray')
        cs = ax.contour(data_smoothed_mag, levels=levels, colors=colors, linewidths=1.6, transform = ax.get_transform(wcs_rebin))
        ax.clabel(cs, cs.levels[::2], fmt='%d mag', fontsize=mpl.rcParams["xtick.labelsize"]) #Old 10 fontsize
        # cbar = plt.colorbar(im, ax = ax)
        # cbar.set_label(r'$\mu_0$ (mag/arcsec$^2$)') 
        # ax.set_xlabel("RA")
        # ax.set_ylabel("Dec")
        ax.coords[0].set_format_unit('deg', decimal=True)
        ax.coords[1].set_format_unit('deg', decimal=True)
        ax.coords[0].set_major_formatter('d.dd')
        ax.coords[1].set_major_formatter('d.dd')

        ax.coords[0].set_axislabel("RA", fontsize=12)
        ax.coords[1].set_axislabel("Dec", fontsize=12)

        for coord in ax.coords:
            coord.set_ticklabel(size=10)
            # coord.set_axislabel(size=12)
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
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection=wcs)

        contour_color_base ="#fffffff2"
        levels=np.arange(25,31.5, 0.5) #(n,m) -> [n,...,m - 1]
        base = mcolors.to_rgb(contour_color_base)
        colors = [tuple(np.clip(c * f, 0, 1) for c in base) for f in np.linspace(1.8, 0.4, len(levels))]
        im = ax.imshow(color_im, origin='lower', cmap='gray')
        cs = ax.contour(data_smoothed_mag, levels=levels, colors=colors, linewidths=1.6, transform = ax.get_transform(wcs_rebin))
        ax.clabel(cs, cs.levels[::2], fmt='%d mag', fontsize=mpl.rcParams["xtick.labelsize"])
        # cbar = plt.colorbar(im, ax = ax)
        # cbar.set_label(r'$\mu_0$ (mag/arcsec$^2$)') 
        # ax.set_xlabel("RA")
        # ax.set_ylabel("Dec")
        ax.coords[0].set_format_unit('deg', decimal=True)
        ax.coords[1].set_format_unit('deg', decimal=True)
        ax.coords[0].set_major_formatter('d.dd')
        ax.coords[1].set_major_formatter('d.dd')

        ax.coords[0].set_axislabel("RA", fontsize=12)
        ax.coords[1].set_axislabel("Dec", fontsize=12)
        for coord in ax.coords:
            coord.set_ticklabel(size=10)
            # coord.set_axislabel(size=12)
        ax.set_xlim(x_bcg - radius, x_bcg + radius)
        ax.set_ylim(y_bcg - radius, y_bcg + radius)
        # plt.xlim(x_bcg - radius, x_bcg + radius)
        # plt.ylim(y_bcg - radius, y_bcg + radius)
        ax.set_title(fr"{cln} ${band}$-Band SB Contours, $\mu_{{{band}}}^{{\rm \text{{lim}}}}$ = {sb_limit:.1f}")
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_fullres_irg_cropped_final.png', dpi=1000, bbox_inches='tight')
        plt.close()



#========================================================================================================================


def integrate(y, x):
    """
    Integration helper function. Integrates y over x 
    
    Parameters:
    -----------
    y, x : arrays

    Returns:
    --------
    sum : float
        Value of the integral
    """

    return simpson(y, x)

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

#Get redshift
ned_z = info_df['redshift'][0]

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
    
#draw contours over an irg color image
coadd_irg_data = np.dstack((
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


# Converting limiting SB in flux units. counts/px
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


# Now we take it profile-by-profile to decide the summing limit. WRONG METHOD
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
sma = np.asarray(int_df['g']['sma_pix'])






# CORRECT METHOD: FINDING THE TOTAL SATELLITE (CLUSTER MEMBER) LIGHT


sex_cats_dict = {}
sex_path = "SExtractor"

for band in bands:


    # Extract column names from header
    col_names = []
    with open(f"{sex_path}/star_seg_catalog_{band}.cat", "r") as f:
        for line in f:
            if line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    col_names.append(parts[2])
            else:
                break  # stop once data starts

    # Read the data
    df = pd.read_csv(
        f"{sex_path}/star_seg_catalog_{band}.cat",
        delim_whitespace=True,
        comment="#",
        names=col_names
    )

    sex_cats_dict[band] = df


# WCS (World Coordinate System) is same for all griz bands, so just using r 
wcs = WCS(coadd_header)

# Add RA, DEC columns with wcs
for band in bands:
    sex_cats_dict[band]["RA"],  sex_cats_dict[band]["DEC"] = wcs.pixel_to_world_values(sex_cats_dict[band]["X_IMAGE"], sex_cats_dict[band]["Y_IMAGE"])



# read in red sequence galaxy catalog

source_redshift_path = f"~/data/Clusters/gen3_processing/{cln}/red_sequence_output/rs_catalog.csv"

needed_columns = [
    'ra', 'dec',  
    'z_phot', 'z_median', 'z_mode', 'z_ml'             
]

df = pd.read_csv(source_redshift_path, 
                 usecols=needed_columns,
                 dtype={
                     'ra': 'float32',
                     'dec': 'float32',
                     'z_phot': 'float32',
                     'z_median': 'float32',
                     'z_mode': 'float32',
                     'z_ml': 'float32'
                 })


# Filter for sources within 1000 arcsec of BCG

center = SkyCoord(ra=info_df['bcg_RA_deg'], dec=info_df['bcg_DEC_deg'], unit='deg', frame='icrs')

sources = SkyCoord(ra=df['ra'].values, 
                   dec=df['dec'].values, 
                   unit='deg', frame='icrs')

separations = center.separation(sources)

sep_arcsec = separations.to(u.arcsec).value

within_cut = sep_arcsec <= 1000

df_near = df[within_cut].copy()

print(f"Total sources in catalog : {len(df)}")
print(f"Sources within {1000} arcsec: {len(df_near)}")
print(f"Kept {len(df_near)/len(df)*100:.1f}% of the catalog")


# Taking z_phot as the source redshift.

# Filtering for membership
delta_z_cut = 0.03

delta_z = np.abs(df_near['z_median'] - ned_z)

z_member = delta_z < delta_z_cut
print(z_member.value_counts())

df_members = df_near[z_member].copy()

print(f"Cluster redshift: z = {ned_z}")
print(f"Redshift cut used: |z_median - z_cluster| < {delta_z_cut}")
print(f"Number of member candidates within 1000\": {len(df_members)}")




# match the common sources between the two catalogs

members_coords = SkyCoord(ra=df_members['ra'].values, 
                          dec=df_members['dec'].values, 
                          unit='deg')

# store matched results per band
matched_flux_dict = {}
match_stats = {}
satellite_light_dict = {}

for band in bands:
    sex_cat = sex_cats_dict[band]
    
    # Create coordinates for this band's catalog
    flux_coords = SkyCoord(ra=sex_cat['RA'].values, 
                           dec=sex_cat['DEC'].values, 
                           unit='deg')
    
    # Cross-match: find closest source in this band's catalog for each member
    idx, sep2d, _ = members_coords.match_to_catalog_sky(flux_coords)
    
    # filter close match
    max_sep = 2.0   # arcsec
    good_match = sep2d.to(u.arcsec).value < max_sep


    # Calculate separation of each matched source from the BCG center
    # matched_coords = members_coords[good_match]
    sep_from_bcg = members_coords.separation(center)
    is_bcg = sep_from_bcg.to(u.arcsec).value < 10.0   # trying 10 arcsec 

    final_mask = good_match & ~is_bcg

    # Build matched dataframe for this band
    matched = df_members[final_mask].copy().reset_index(drop=True)
    
    # Add the flux from this band's catalog
    matched['FLUX_ISO'] = sex_cat.iloc[idx[final_mask]]['FLUX_ISO'].values
    
    # Optional: add separation and other useful columns
    matched['separation_arcsec'] = sep2d.to(u.arcsec).value[final_mask]
    matched['separation_from_bcg_arcsec'] = sep_from_bcg.to(u.arcsec).value[final_mask]
    # matched['match_index'] = idx[good_match]
    
    matched_flux_dict[band] = matched

    # Sum satellite light (this is what you will add to bcg_icl_light)
    satellite_light_dict[band] = matched['FLUX_ISO'].sum()
    
    # Save some stats
    match_stats[band] = {
        'total_members': len(df_members),
        'matched': good_match.sum(),
        'match_rate': good_match.mean() * 100,
        'mean_sep': matched['separation_arcsec'].mean()
    }
    
    print(f"{band}: Matched {final_mask.sum()} satellites | "
          f"Satellite flux sum = {satellite_light_dict[band]:.2f} | "
          f"Excluded {is_bcg.sum()} BCG candidate(s)")



total_light_dict = {}
total_light_err_dict = {}
bcg_icl_light_dict = {}
bcg_icl_light_err_dict = {}
icl_light_dict = {}
icl_light_err_dict = {}
bcg_light_dict = {}
bcg_light_err_dict = {}

universal_radius_px = mpc_to_px

for band in bands:

    x_mask = sma <= universal_radius_px
    x_cut = sma[x_mask]
    
    print(f"{band}: integrating everything to {x_cut[-1]:.1f} px ({universal_radius_px:.1f} px = 1 Mpc)")


    
    # total_cut   = total_prof_dict[band][x_mask]
    bcg_icl_cut = bcg_icl_prof_dict[band][x_mask]
    bcg_cut     = bcg_prof_dict[band][x_mask]
    icl_cut     = icl_prof_dict[band][x_mask]

    # total_err_cut   = total_prof_err_dict[band][x_mask]
    bcg_icl_err_cut = bcg_icl_prof_err_dict[band][x_mask]
    bcg_err_cut     = bcg_prof_err_dict[band][x_mask]
    icl_err_cut     = icl_prof_err_dict[band][x_mask]

    # Prevent negative ICL flux values?
    icl_cut = np.maximum(icl_cut, 0.0)


    # total_light_dict[band] = integrate(total_cut, x_cut)
    bcg_icl_light_dict[band] = integrate(bcg_icl_cut*x_cut, x_cut)*2*np.pi
    bcg_light_dict[band] = integrate(bcg_cut*x_cut, x_cut)*2*np.pi
    icl_light_dict[band] = integrate(icl_cut*x_cut, x_cut)*2*np.pi      

    # Monte Carlo error prop
    n_mc = 500
    areas_total = []
    areas_bcg_icl = []
    areas_bcg = []
    areas_icl = []


    for _ in range(n_mc):
        # sample_total_flux = total_cut + np.random.normal(0, total_err_cut)
        # areas_total.append(integrate(sample_total_flux, x=x_cut))

        sample_bcg_icl_flux = bcg_icl_cut + np.random.normal(0, bcg_icl_err_cut)
        areas_bcg_icl.append(integrate(sample_bcg_icl_flux*x_cut, x=x_cut)*2*np.pi)

        sample_bcg_flux = bcg_cut + np.random.normal(0, bcg_err_cut)
        areas_bcg.append(integrate(sample_bcg_flux*x_cut, x=x_cut)*2*np.pi)

        sample_icl_flux = icl_cut + np.random.normal(0, icl_err_cut)
        areas_icl.append(integrate(sample_icl_flux*x_cut, x=x_cut)*2*np.pi)

    # total_light_err_dict[band] =  np.std(areas_total)
    bcg_icl_light_err_dict[band] =  np.std(areas_bcg_icl)
    bcg_light_err_dict[band] =  np.std(areas_bcg)
    icl_light_err_dict[band] =  np.std(areas_icl)


total_light_dict = {}

for band in bands:
    total_light_dict[band] = bcg_icl_light_dict[band] + satellite_light_dict.get(band, 0.0)


print(f"{band} BCG+ICL: {bcg_icl_light_dict[band]:.2f} | "
          f"Satellites: {satellite_light_dict.get(band, 0.0):.2f} | "
          f"Total cluster light: {total_light_dict[band]:.2f}")


total_light_err_dict = {band: 0.0 for band in bands}







# Now take the ratios and propagate errors!

bcg_icl_frac_dict = {band: bcg_icl_light_dict[band]/total_light_dict[band] for band in bands}
bcg_frac_dict = {band: bcg_light_dict[band]/total_light_dict[band] for band in bands}
icl_frac_dict = {band: icl_light_dict[band]/total_light_dict[band] for band in bands}
icl_over_bcgicl_frac_dict = {band: icl_light_dict[band]/bcg_icl_light_dict[band] for band in bands}

bcg_icl_frac_err_dict = {band: np.sqrt((bcg_icl_frac_dict[band]**2)*
    ((bcg_icl_light_err_dict[band]/bcg_icl_light_dict[band])**2 +
    (total_light_err_dict[band]/total_light_dict[band])**2)) for band in bands}

bcg_frac_err_dict = {band: np.sqrt((bcg_frac_dict[band]**2)*
    ((bcg_light_err_dict[band]/bcg_light_dict[band])**2 +
    (total_light_err_dict[band]/total_light_dict[band])**2)) for band in bands}

icl_frac_err_dict = {band: np.sqrt((icl_frac_dict[band]**2)*
    ((icl_light_err_dict[band]/icl_light_dict[band])**2 +
    (total_light_err_dict[band]/total_light_dict[band])**2)) for band in bands}

icl_over_bcgicl_frac_err_dict = {band: np.sqrt((icl_over_bcgicl_frac_dict[band]**2)*
    ((icl_light_err_dict[band]/icl_light_dict[band])**2 +
    (bcg_icl_light_err_dict[band]/bcg_icl_light_dict[band])**2)) for band in bands}


for band in bands:
    print(f"{band} BCG+ICL Fraction: {bcg_icl_frac_dict[band]*100:.2f}% pm  {bcg_icl_frac_err_dict[band]*100:.2f}%")
    print(f"{band} BCG Fraction: {bcg_frac_dict[band]*100:.2f}% pm  {bcg_frac_err_dict[band]*100:.2f}%")
    print(f"{band} ICL Fraction: {icl_frac_dict[band]*100:.2f}% pm  {icl_frac_err_dict[band]*100:.2f}%")
    print(f"{band} ICL/(BCG+ICL) Fraction: {icl_over_bcgicl_frac_dict[band]*100:.2f}% pm  {icl_over_bcgicl_frac_err_dict[band]*100:.2f}%")


# Save into a df/csv

fraction_df = pd.DataFrame({
    "band": bcg_icl_frac_dict.keys(),
    "BCG_ICL_frac": bcg_icl_frac_dict.values(),
    "BCG_frac": bcg_frac_dict.values(),
    "ICL_frac": icl_frac_dict.values(),
    "BCG_ICL_frac_err": bcg_icl_frac_err_dict.values(),
    "BCG_frac_err": bcg_frac_err_dict.values(),
    "ICL_frac_err": icl_frac_err_dict.values(),
    "ICL_over_BCGICL_frac": icl_over_bcgicl_frac_dict.values(),
    "ICL_over_BCGICL_frac_err": icl_over_bcgicl_frac_err_dict.values(),
    "ICL_lum": icl_light_dict.values(),
    "ICL_lum_err": icl_light_err_dict.values(),
    "BCG_lum": bcg_light_dict.values(),
    "BCG_lum_err": bcg_light_err_dict.values(),
    "RS_lum": satellite_light_dict.values(),
    "TOTAL_lum": total_light_dict.values(),
    "TOTAL_lum_err": total_light_err_dict.values()
})

fraction_df.to_csv(f'{output_folder}/{cln}_fractions.csv', index=False)



# fig, ax = plt.subplots(figsize=(8, 6))

# for band in bands:
#     ax.errorbar(sma[band], total_prof_dict[band], yerr=total_prof_err_dict[band], color=color_dict[band], 
#                 fmt='o', capsize=6, markersize=6, mec='black', alpha=0.6, label=fr"${band}$, $\mu_{{lim}}$ = {info_df[f'limiting_mag_{band}'].iloc[0]:.1f}")
    
#     plt.axhline(y=lim_flux_px_dict[band], color=color_dict[band], linestyle='--', linewidth=1) #horizontal line marking over/undersubtraction

# ax.set_xscale('log')
# # ax.set_yscale('log')

# ax.set_xlabel('SMA (px)')
# ax.set_ylabel(r'Intensity (counts/px$^2$)')
# # ax.set_ylim(19, 31) #Fix y limits

# # Top x-axis with kpc
# ax_top = ax.twiny()
# ax_top.set_xscale('log')
# ax_top.set_xlim(int_df['g']['sma_kpc'].iloc[0], int_df['g']['sma_kpc'].iloc[-1])
# ax_top.set_xlabel('SMA (kpc)')

# # ax.invert_yaxis()
# ax.tick_params(bottom=True, top=False, left=True, right=True)
# ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
# ax.tick_params(direction="in")
# ax.legend(prop={'size': 12})
# ax.set_title(f'{cln} Final ICL Profiles', fontsize = 15)
# plt.tight_layout()
# # plt.show()
# plt.savefig(f'{output_folder}/total_light_radial_curve.png', dpi=1000, bbox_inches='tight')
# plt.close()