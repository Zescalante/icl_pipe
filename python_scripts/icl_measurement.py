

#========================================================================================================================
# Import libraries

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
from photutils.segmentation import SegmentationImage
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
plt.rcParams['figure.dpi'] = 1000

import os
import pandas as pd
import pdfplumber
import re
import subprocess 
import sys
import tempfile
import urllib.request



#========================================================================================================================
#========================================================================================================================
# CONSTANTS
#========================================================================================================================
#========================================================================================================================

rad_to_arcsec =  206264.806   # 1/1^{''} #arcsec per rad
arcsec_per_pix_decam = 0.263 


#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=14:
    print("Usage: python this_script.py cln refcat coadd_g coadd_r coadd_i coadd_z lsp_mask_g lsp_mask_r lsp_mask_i lsp_mask_z input_bcg_csv output_folder sextractor_folder") 
    sys.exit(1)

cln = sys.argv[1]
refcat = sys.argv[2]
coadd_g_fits = sys.argv[3]
coadd_r_fits = sys.argv[4]
coadd_i_fits = sys.argv[5]
coadd_z_fits = sys.argv[6]
lsp_mask_g_fits = sys.argv[7]
lsp_mask_r_fits = sys.argv[8]
lsp_mask_i_fits = sys.argv[9]
lsp_mask_z_fits = sys.argv[10]
bcg_coords_csv_name = sys.argv[11]
bcg_coords_df = pd.read_csv(bcg_coords_csv_name)
output_folder = sys.argv[12]
sextractor_folder = sys.argv[13]

# Below path is to get the errors for coadds
photometric_output_folder = f'/gpfs/data/idellant/Clusters/gen3_processing/{cln}/photometric_correction_output' # Update later

band_list = ['g','r','i','z']
color_dict = {'g': 'dodgerblue', 'r': 'tomato', 'i': 'purple', 'z': 'goldenrod'}  #Band color scheme
# coadd_fits_dict = {'g': coadd_g_fits, 'r': coadd_r_fits, 'i': coadd_i_fits, 'z': coadd_z_fits}


#========================================================================================================================
#========================================================================================================================
# FUNCTIONS
#========================================================================================================================
#========================================================================================================================

def annulus_flux_measure(coadd, iso_in, iso_out):
    """
    Measure integrated flux inside ellipse annuli.

    Parameters:
    -----------
    coadd : 2D numpy array
        The input masked image.
    iso_in : Isophote object
        object containing info about inner isophote.
    iso_out : Isophote object
        object containing info about outer isophote.

    Returns:
    --------
    annulus_flux : float
        Pixel count/intensity within the annulus.
    full_mask : 2D numpy array
        Array of annuli pixel indices. Same shape as coadd
    """

    # Get isophote center. Same for inner and outer
    x0 = iso_in.x0
    y0 = iso_in.y0

    # Size of coadd
    ny,nx = coadd.shape
    
    # Semi-major axes
    sma_in = iso_in.sma
    sma_out = iso_out.sma

    # Generatie ellipses from isophote parameters
    ep_geom_in = EllipseGeometry(x0, y0, sma_in, iso_in.eps, iso_in.pa)
    ep_geom_out = EllipseGeometry(x0, y0, sma_out, iso_out.eps, iso_out.pa)

    pad = 10 #Original padding of 1

    # Create bounding box from outer ellipse sma
    imin = max(0, int(x0 - sma_out - 0.5) - pad)
    imax = min(nx, int(x0 + sma_out + 0.5) + pad)
    jmin = max(0, int(y0 - sma_out - 0.5) - pad)
    jmax = min(ny, int(y0 + sma_out + 0.5) + pad)

    y, x = np.mgrid[jmin:jmax, imin:imax]

    # Radius and angle (same for both isophotes)
    r_in, angle = ep_geom_in.to_polar(x, y)
    r_out, _ = ep_geom_out.to_polar(x, y)

    r_in_boundary = ep_geom_in.radius(angle)
    r_out_boundary = ep_geom_out.radius(angle)
    
    # Find the annulus pixel mask and sum
    annulus_mask = (r_in > r_in_boundary) & (r_out <= r_out_boundary)
    annulus_flux = np.ma.sum(coadd[y[annulus_mask], x[annulus_mask]]) #ignores masked pixels

    # Convert mask to full mask same size as coadd
    full_mask = np.zeros_like(coadd, dtype=bool)
    full_mask[jmin:jmax, imin:imax] = annulus_mask

    return annulus_flux, full_mask

#------------------------------------------------------------------------------------------------------------------------

def apply_lsp_mask(coadd, mask_arr, y_bcg, x_bcg, band, header, save_full_mask_im=True, unmsk_scale = 100, run_num=1):
    """
    Apply lsp mask to it's coadd.
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input array.
    mask_arr : 2D numpy array
        The input mask.
    bcg_val : int
        Value in mask_arr to leave unmasked.
    band : str
        Specify which band (e.g. "r", "i")
    header : fits header
        fits header information (WCS) to apply to new image
    save_full_mask_png : boolean
        Set whether to save the fully masked png. Default is False
    unmsk_scale : int
        Physical scale (kpc) for which to unmask, centered on BCG.
    run_num : int
        Iteration value. Default 1.
        
    Returns:
    --------
    masked array object
        The masked coadd image
    """

    # convert physical scale radius to pix
    radius = int(unmsk_scale/size_kpc)

    # Get size of image
    ny, nx = coadd.shape
    
    # Define bounding box
    x_min = max(0, x_bcg - radius)
    x_max = min(nx, x_bcg + radius + 1)
    y_min = max(0, y_bcg - radius)
    y_max = min(ny, y_bcg + radius + 1)
    
    # Extract sub-image
    sub_img = coadd[y_min:y_max, x_min:x_max]

    # Create distance mask
    yy, xx = np.indices(sub_img.shape)
    dx = xx + x_min - x_bcg
    dy = yy + y_min - y_bcg
    mask = dx**2 + dy**2 <= radius**2

    region_mask_fullsize = np.zeros_like(mask_arr, dtype=bool)
    # region_mask_fullsize[y_min:y_max, x_min:x_max] = mask #if you want a circular region mask
    region_mask_fullsize[y_min:y_max, x_min:x_max] = True #if you want a square region mask


    # Unmask region centered on BCG
    # mask_arr_center_unmsk = np.where(mask, 0, mask_arr)
    mask_arr_center_unmsk = mask_arr.copy()
    mask_arr_center_unmsk[region_mask_fullsize] = 0

    # Where mask_arr equals bcg_val, replace value with 0
    # arr = np.where(mask_arr == bcg_val,0,mask_arr)

    # Where arr is not zero, replace with 1
    # arr = np.where(arr != 0,1, arr)

    # if save_full_mask_im:

    #     plt.figure()
    #     cmap = plt.cm.viridis.copy() 
    #     # cmap.set_bad(color='black')#Setting masked pixels to black
    #     cmap.set_under(cmap(0))
    #     plt.imshow((display(ma.masked_array(coadd, mask=mask_arr)).filled(fill_value=-1))*255, cmap=cmap)
    #     plt.gca().invert_yaxis()
    #     plt.colorbar(label="Intensity", orientation="vertical") 
    #     # plt.xlim(x_bcg_ref - 500, x_bcg_ref + 500)
    #     # plt.ylim(y_bcg_ref - 500, y_bcg_ref + 500)
    #     plt.xlabel('x (pixels)')
    #     plt.ylabel('y (pixels)')
    #     plt.title(f'{cln} Fully-Masked {band}-band Coadd')
    #     plt.savefig(f'{output_folder}/{cln}_{band}_coadd_fully_masked_iter_{run_num}.png', dpi=1000, bbox_inches='tight')
    #     plt.close()

    #     # Save fully masked image as fits. Does not work atm
    #     # hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=np.where(arr != 0,1, mask_arr)).filled(np.nan), header=header)
    #     hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=mask_arr).filled(np.nan), header=header)
    #     hdu.writeto(f'{output_folder}/fully_masked_{band}_iter_{run_num}.fits', overwrite=True)

    # coadd cutout to run sextractor on
    coadd_cutout = coadd[y_min:y_max, x_min:x_max]
    hdu = fits.PrimaryHDU(coadd_cutout, header=None)
    hdu.writeto(f'{output_folder}/coadd_region_cutout_{band}_iter_{run_num}.fits', overwrite=True)

    
    # Apply binary mask to coadd image and return. Value 1 means mask
    return ma.masked_array(coadd, mask=mask_arr_center_unmsk)


#------------------------------------------------------------------------------------------------------------------------

def create_new_segmap(coadd, orig_segmap_data, x_center, y_center, band, header, radius = 50, run_num = 1):
    """
    Remove the closet mask (pixel-wise) to the given coordinates (usually BCG coords)
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input image.
    orig_segmap_data: 2D numpy array
        Image data of original segmentation map.
    x_center, y_center : float
        Center to search from.
    band : string
        String label. (e.g. "r", "i")
    header : fits header 
        Header information (WCS) to save to new image.
    radius : float
        Search radius in pixels. Default 50 pix.
    run_num: int
        Iteration number. Default value 1
        
    Returns:
    --------
    combined_seg_data : 2D numpy array
        np array of the new, combined segmentation image.
    """

    # Convert sersic diff image array to fits
    hdu = fits.PrimaryHDU(coadd)

    # Save sersic diff fits. Modified run_sextractor so I no longer need to save this
    # hdu.writeto(f'icl_measurement_output/{cln}_{band}_sersic_diff_{run_num}.fits', overwrite=True)

    # Run SExtractor on image fits
    seg_map_fits = run_sextractor(hdu, band=band, run_num=run_num, detect_thresh_param='5.0')

    # Load new seg map data
    seg_map_data = seg_map_fits[0].data 

    # Wrap seg map data into SegmentationImage
    seg_image = SegmentationImage(seg_map_data) 

    # Build a grid of pixel coords
    yy, xx = np.indices(seg_map_data.shape)

    # Compute distance from each pixel to center
    dist = np.hypot(xx - x_center, yy - y_center)

    # Mask: only consider pixels within radius
    within = (dist <= radius) & (seg_map_data > 0)

    if np.any(within):

        # Commented out code is to remove closest mask to center within radius

        # # Find the pixel closest to center within radius
        # idx = np.argmin(dist[within])

        # # Get corresponding label
        # labels_within = seg_map_data[within]
        # remove_id = labels_within.flat[idx]

        # print(f"Removing label {remove_id}")


        # If you want to remove all masks within radius
        labels_to_remove = np.unique(seg_map_data[within])
        labels_to_remove = labels_to_remove[labels_to_remove > 0]  # drop background

        segm_removed = seg_image.copy()
        segm_removed.remove_labels(labels_to_remove)

        # Save the cleaned segmap as fits 
        # fits.writeto(f"{sextractor_folder}/seg_{band}_{run_num}_cleaned.fits", segm_removed.data, overwrite=True)

    else:
        print("No labeled pixel found within radius.")
        segm_removed = seg_image.copy()

    # Convert cleaned seg map back to array
    segm_removed = segm_removed.data

    #Combine seg map data values
    combined_seg_data = orig_segmap_data + segm_removed 

    #convert combined seg map to fits (might need to add WCS back to this)
    hdu_2 = fits.PrimaryHDU(combined_seg_data, header=header)
    hdu_2.writeto(f'{sextractor_folder}/seg_{band}_{run_num}.fits', overwrite=True) #Save combined seg map fits to directory
    
    return combined_seg_data

#------------------------------------------------------------------------------------------------------------------------

def display(mat, vmin=-0.1, vmax=250):
    """
    Clip the intensity and stretch the data before displaying it.
    
    Parameters:
    -----------
    mat : 2D numpy array
        The input image.
    vmin, vmax : float
        Clipping bounds. Sets display range. Default range [-0.1, 250].
        
    Returns:
    --------
    mat_2 : 2D numpy array
        Clipped and stretched image
    """

    interval = ManualInterval(vmin, vmax)
    mat_1 = interval(mat)
    stretch = AsinhStretch(0.0004)
    mat_2 = stretch(mat_1)

    return mat_2

#------------------------------------------------------------------------------------------------------------------------

def draw_lsb_contours(coadd, band, superpix_scale=16, sigma=3, block_method=np.mean):
    """
    Draw magnitude contours on smoothed coadd.
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input array/image.
    band : string
        String label. (e.g. "r", "i")
    superpix_scale : int
        Superpixel binning factor. Default 16 pix.
    sigma : int
        Standard deviation value to use for gaussian kernel smoothing. Default 3 sigma.
    block_method : func
        Method to combine pixels. Default is mean.
        
    Returns:
    --------
    mat_2 : 2D numpy array
        Clipped and stretched image
    """
    arcsec_per_pix_super = superpix_scale*arcsec_per_pix_decam

    # Separate masked array
    data = coadd.data
    mask = coadd.mask.astype(bool)

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



    # # change pixel scale, block and smooth
    # arcsec_per_pix_super = superpix_scale*arcsec_per_pix_decam
    # superpix_coadd = block_reduce(coadd, block_size=superpix_scale, func=block_method)

    # smoothed_coadd = gaussian_filter(superpix_coadd, sigma=sigma)

    # smoothed_coadd_arcsec = smoothed_coadd/(arcsec_per_pix_super)**2
    # smoothed_coadd_mag = -2.5*np.log10(smoothed_coadd_arcsec) + 27

    # smoothed_coadd_mag = np.where(smoothed_coadd_mag > 0,
    #                          smoothed_coadd_mag,
    #                          np.nan)

    # now we need contours

    plt.figure()
    # plt.imshow(data_smoothed_mag, origin='lower', cmap='gray', vmin=1, vmax=30)
    # plt.imshow(data_smoothed_mag, origin='lower', cmap='gray')
    # plt.imshow(display(data_smoothed)*255, origin='lower', cmap='gray')
    plt.imshow(1 - display(data_smoothed), origin='lower', cmap='gray')
    cs = plt.contour(data_smoothed_mag, levels=[26,27,28,29,30,31,32], colors=['black','purple','blue','green','orange','red','indianred'], linewidths=1.2)
    plt.clabel(cs, fmt='%d mag', fontsize=6)
    plt.colorbar(label=r'$\mu_0$ (mag/arcsec$^2$)', orientation="vertical") 
    # plt.imshow(display(smoothed_coadd)*255)
    # plt.gca().invert_yaxis()
    # plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    # plt.xlim(x_bcg_ref - 1000, x_bcg_ref + 1000)
    # plt.ylim(y_bcg_ref - 1000, y_bcg_ref + 1000)
    plt.title(f'{cln} {band}-Band SB Contours')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()




#------------------------------------------------------------------------------------------------------------------------
# DONT THINK I NEED THIS FUNCTION ANYMORE BUT KEEPING IT AROUND IN CASE
def fit_ellip_annuli(band, coadd, center, errors, spacings=np.linspace(10, 2900, 40), theta=45, run_num=1):

    """
    Fit elliptical annuli, measure flux within annuli.
    
    Parameters:
    -----------
    band : string
        String label. (e.g. "r", "i")
    coadd : 2D numpy array
        The input array/image.
    center : tuple
        (x0, y0) that specifies center to fit annuli.
    errors: 2D numpy array
        Error array for input coadd image.
    spacings : 1D array of radii. Defines edges of annuli.
               Default is 1D array of even spacings, 30 values between 10 to 1900 pix.
    theta : float. 
        Position angle of ellipses in degrees. Default 45 deg.
    run_num: int
        Iteration number. Default value 1
        
    Returns:
    --------
    sums : list
        List of counts/intensities within each radius.
    sums_err : list
        List of intensity errors.
    areas: list
        List of areas (pix) of each elliptical annulus.
    """

    # Array to hold summed intensities in each annulus
    sums = []
    # Intensity sum errors
    sums_err = []
    # Array to hold annulus areas
    areas = []

    # Plot coadd
    plt.figure(figsize=(10,8))
    plt.imshow(display(coadd)*255)

    # Begin fitting annuli. Append values to empty lists
    for ind in range(len(spacings) - 1):
        annulus = EllipticalAnnulus(center, a_in=spacings[ind], a_out=spacings[ind+1], \
                               b_out=spacings[ind], theta=theta)
        annulus.plot(color='red', lw=1)

        photometry = annulus.do_photometry(coadd, errors)

        sums.append(list(photometry[0]))
        sums_err.append(list(photometry[1]))
        # aperstats = ApertureStats(data, annulus_aperture)
        areas.append(annulus.area)

    # Convert list of lists to list
    sums = sum(sums, [])
    sums_err = sum(sums_err, [])

    # print(sums)
    # print(sums_err)

    # Other plot parameters
    plt.gca().invert_yaxis()
    plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlim(center[0] - 3000, center[0] + 3000) #For original linspace range 
    plt.ylim(center[1] - 3000, center[1] + 3000) #Might need to update boundary
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title(f'{cln} {band} Band Elliptical Annuli')
    plt.savefig(f'{output_folder}/{cln}_{band}_background_annuli_iter_{run_num}.png', dpi=1000, bbox_inches='tight')
    plt.close()

    # Return annuli sums (units of counts), their uncertainties, and areas (units of pixels)
    return sums, sums_err, areas

#------------------------------------------------------------------------------------------------------------------------

def get_photometric_errors(file_path=photometric_output_folder, cat=refcat, bands = band_list):
    """
    Retrieve errors from photometric correction magnitude difference csv
    
    Parameters:
    -----------
    file_path : str
        Location of difference csv files.
    cat : str
        Name of reference catalog.
    bands : str
        List of bands. Default band_list.
        
    Returns:
    --------
    List
        List of input error floats for coadds, in units of magnitude.
    """

    output_errors = []

    for band in bands:
        df = pd.read_csv(f'{file_path}/{cln}_mag_diffs_{cat}.csv')

        diff = df[f'{band}_diff'] #Difference values 
        diff = pd.to_numeric(diff, errors='coerce')  # convert blanks to NaN

        # Generating same bins and midpoints as carried out in photometric_correction
        bins = np.linspace(-0.25, 0.25, 50*2+1)

        # n, b, p = axs[1].hist(diff, bins=bins, histtype="step", log=True, label=band) # Code from compare_mag_v3b.py
        n, edges = np.histogram(diff, bins=bins) # Same as above but without plotting

        mid_points = (edges[1:] + edges[:-1])/2.

        # Find max and half-max
        max_val = np.max(n)
        half_max = max_val/2.0

        # Discrete calculation
        indices = np.where(n >= half_max)[0] #indices where value is greater than half maximum
        if len(indices) >= 2: 
            left_idx, right_idx = indices[0], indices[-1] #First and last values (boundaries)
            fwhm = mid_points[right_idx] - mid_points[left_idx]
            hwhm = fwhm / 2.0
            # center = mid_points[np.argmax(n)]
            output_errors.append(hwhm)
            # output_errors[band] = [hwhm]

        else:
            print("No valid half-maximum crossings.")

        # Interpolation method
        # interp = interp1d(mid_points, n, kind='linear', fill_value="extrapolate")
        # fine_x = np.linspace(mid_points.min(), mid_points.max(), 1000)
        # fine_y = interp(fine_x) # Interpolated magnitude values

        # half_max = fine_y.max() / 2
        # mask = fine_y >= half_max

        # if np.any(mask):
        #     left_x, right_x = fine_x[mask][0], fine_x[mask][-1]
        #     fwhm = right_x - left_x
        #     hwhm = fwhm / 2

        #     output_errors.append(hwhm)
        
    return output_errors

#------------------------------------------------------------------------------------------------------------------------
        
def isophote_fitter(coadd, x0, y0, max_SMA):
    """
    Fits isophotes to input masked-coadd, trying different starting position angles.
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input masked array/image.
    x0, y0 : float
        Center values from which to fit isophotes.
    max_SMA : int
        max sma radius 

    Returns:
    --------
    returns an IsophoteList object
        object containing info about each fit isophote.
    """

    pa_list = [np.pi/12, np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4, 7*np.pi/8, 11*np.pi/12] #Starting PAs to try
    min_sma_list = [10, 20, 40, 60, 80, 100, 110, 120, 140] #Minimum smas to try (pix)

    for start_sma in min_sma_list:

        print(f"Trying fit with min sma = {start_sma} pix")
        try:

            for start_pa in pa_list:
                try:
                    # Starting ellipse geometry
                    el_geom = EllipseGeometry(x0=int(x0), y0=int(y0), \
                                sma=5, eps=0.1, pa=start_pa) #Originally used pa=0
                    
                    # This is performed on the masked coadds
                    el_fixed = Ellipse(coadd, el_geom) 
                    
                    # Fits the isophotes. This returns an IsophoteList object
                    isophotes = el_fixed.fit_image(sma0=start_sma + 5, minsma=start_sma, maxsma=max_SMA, step=0.12, fix_center=True, \
                                            fix_pa=False, fix_eps=False) #minsma = 10 default
                    
                    if not isophotes or len(isophotes) == 0:
                        print(f"No valid isophotes for PA = {start_pa:.2f} rad, retrying...")
                        continue
                    
                    print(f"Fit succeeded with PA = {start_pa:.2f} rad and min sma = {start_sma} pix")
                    return isophotes
                
                except Exception as e:
                    print(f"Error occurred during isophote-fitting process (PA loop)")
                    continue

            # raise RuntimeError(f"Isophote fitting failed for all PA attempts")
        
        except Exception as e:
            print(f"Error occurred during isophote-fitting process (sma loop)")
            continue
    
    raise RuntimeError("Isophote fitting failed for all SMA and PA attempts")

#------------------------------------------------------------------------------------------------------------------------

def mask_single_coadd(coadd, mask_arr, bcg_val, band, header, save_full_mask_im=True, run_num=1):
    """
    Unmask BCG in coadd image, and (ideally) mask everything else.
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input array.
    mask_arr : 2D numpy array
        The input mask.
    bcg_val : int
        Value in mask_arr to leave unmasked.
    band : str
        Specify which band (e.g. "r", "i")
    header : fits header
        fits header information (WCS) to apply to new image
    save_full_mask_png : boolean
        Set whether to save the fully masked png. Default is False
    run_num : int
        Iteration value. Default 1.
        
    Returns:
    --------
    masked array object
        The masked coadd image
    """

    # Where mask_arr equals bcg_val, replace value with 0
    arr = np.where(mask_arr == bcg_val,0,mask_arr)

    # Where arr is not zero, replace with 1
    arr = np.where(arr != 0,1, arr)

    if save_full_mask_im:

        plt.figure()
        cmap = plt.cm.viridis.copy() 
        # cmap.set_bad(color='black')#Setting masked pixels to black
        cmap.set_under(cmap(0))
        plt.imshow((display(ma.masked_array(coadd, mask=mask_arr)).filled(fill_value=-1))*255, cmap=cmap)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        # plt.xlim(x_bcg_ref - 500, x_bcg_ref + 500)
        # plt.ylim(y_bcg_ref - 500, y_bcg_ref + 500)
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.title(f'{cln} Fully-Masked {band}-band Coadd')
        plt.savefig(f'{output_folder}/{cln}_{band}_coadd_fully_masked_iter_{run_num}.png', dpi=1000, bbox_inches='tight')
        plt.close()

        # Save fully masked image as fits. Does not work atm
        # hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=np.where(arr != 0,1, mask_arr)).filled(np.nan), header=header)
        hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=mask_arr).filled(np.nan), header=header)
        hdu.writeto(f'{output_folder}/fully_masked_{band}_iter_{run_num}.fits', overwrite=True)

    
    # Apply binary mask to coadd image and return. Value 1 means mask
    return ma.masked_array(coadd, mask=arr)

#------------------------------------------------------------------------------------------------------------------------

def recenter_bcg_coords(coadd, x_bcg, y_bcg, radius=25):
    """
    Find the brightest pixel within a radius of a given (x, y) guess.
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input image.
    x_bcg, y_bcg : float
        Initial guess for the coordinates.
    radius : float
        Search radius in pixels. Default 25 pix.
        
    Returns:
    --------
    (x_max, y_max) : (int, int) tuple
        Coordinates of the brightest pixel found within the radius.
    """

    # Get size of image
    ny, nx = coadd.shape

    # Convert input coordinates to integer values
    x_bcg = int(np.round(x_bcg).item())
    y_bcg = int(np.round(y_bcg).item())
    
    # Define bounding box for the search area
    x_min = max(0, x_bcg - radius)
    x_max = min(nx, x_bcg + radius + 1)
    y_min = max(0, y_bcg - radius)
    y_max = min(ny, y_bcg + radius + 1)
    
    # Extract sub-image
    sub_img = coadd[y_min:y_max, x_min:x_max]

    # Create distance mask
    yy, xx = np.indices(sub_img.shape)
    dx = xx + x_min - x_bcg
    dy = yy + y_min - y_bcg
    mask = dx**2 + dy**2 <= radius**2

    # Apply mask
    masked_img = np.where(mask, sub_img, -np.inf)

    # Find max pixel within the mask
    max_idx = np.unravel_index(np.argmax(masked_img), masked_img.shape)
    y_max_pixel = y_min + max_idx[0]
    x_max_pixel = x_min + max_idx[1]
    
    return x_max_pixel, y_max_pixel

#------------------------------------------------------------------------------------------------------------------------

def run_sextractor(input_fits, 
                   band,
                   run_num=1,
                   config_file=f'{sextractor_folder}/default.sex', 
                   param_file=f'{sextractor_folder}/default.param',
                   filter_file=f'{sextractor_folder}/default.conv',
                #    segmap_name=f'{sextractor_folder}/seg.fits',
                    detect_thresh_param = '1.5',
                    tag='regular'
                   ):
    """
    Runs SExtractor on the input FITS file and returns the path to the segmentation FITS file.

    Parameters
    ----------
    input_fits : str
        Path to the input FITS image, or in-memory fits object
    band : str
        Specify which band (e.g. "r", "i")
    run_num : Which iteration (run) of sextractor
    config_file : str
        Path to the SExtractor config file (default 'SExtractor/default.sex').
    param_file : str
        Path to the SExtractor parameter file (default 'SExtractor/default.param').
    filter_file : str
        Path to the SExtractor convolution filter file (default 'SExtractor/default.conv').
    segmap_name : str
        Filename to save the segmentation image as (default 'seg.fits').
    detect_thresh_param: str
        Sigma value above which is considered detection.
    Returns
    -------
    seg_fits : 
        Segmentation map FITS image.
    cat_df : 
        catalog as a pandas df.
    """
    
    # Filename/path to save the segmentation image and catalog as
    segmap_name = f'{sextractor_folder}/seg_{tag}_{band}_{run_num}.fits'
    cat_name = f'{sextractor_folder}/catalog_{tag}_{band}_{run_num}.cat'

    # Left check from when only accepted fits path
    # if not os.path.exists(input_fits):
    #     raise FileNotFoundError(f"Input file not found: {input_fits}")


    # Accept either path or in-memory data 
    cleanup_temp = False
    if isinstance(input_fits, str):
        # It's a path
        if not os.path.exists(input_fits):
            raise FileNotFoundError(f"Input file not found: {input_fits}")
        input_path = input_fits
    else:
        # It's an in-memory .fits hdu or numpy array, then save temp file
        cleanup_temp = True
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            input_path = tmp.name
        if isinstance(input_fits, fits.PrimaryHDU):
            input_fits.writeto(input_path, overwrite=True)
        elif isinstance(input_fits, fits.HDUList):
            input_fits.writeto(input_path, overwrite=True)
        else:
            # Assume it's a numpy array
            fits.PrimaryHDU(input_fits).writeto(input_path, overwrite=True)



    # Build the SExtractor command
    command = [
        'sex', input_path,
        '-c', config_file,
        '-CHECKIMAGE_TYPE', 'SEGMENTATION',
        '-CHECKIMAGE_NAME', segmap_name,
        # '-CATALOG_NAME', '/dev/null',  # Suppress catalog output if not needed
        '-CATALOG_NAME', cat_name,
        '-PARAMETERS_NAME', param_file, 
        '-FILTER_NAME', filter_file,
        '-DETECT_THRESH', detect_thresh_param #Higher val means smaller, less agressive masks
    ]

    # Run the command
    subprocess.run(command, 
                   stdout=subprocess.DEVNULL,   # hide normal messages
                   stderr=subprocess.DEVNULL,   # hide warnings
                   check=True)

    # Return the path to the segmentation map
    if not os.path.exists(segmap_name):
        raise RuntimeError(f"SExtractor did not produce a segmentation map for {band} band.")
    if not os.path.exists(cat_name):
        raise RuntimeError(f"SExtractor did not produce a catalog for {band} band.")


    # Load the segmentation file
    seg_fits = fits.open(segmap_name)

    # Attach WCS/header from the input fits
    if isinstance(input_fits, str):
        input_header = fits.getheader(input_fits)
    else:
        if isinstance(input_fits, fits.PrimaryHDU):
            input_header = input_fits.header
        elif isinstance(input_fits, fits.HDUList):
            input_header = input_fits[0].header
        else:
            input_header = None

    # Attach header to segmap
    if input_header is not None:
        seg_fits[0].header.update(input_header)
        seg_fits.writeto(segmap_name, overwrite=True)


    # Below is code for loading the output sextractor catalog

    # # Grabbing the column names from catalog
    # colnames = []
    # with open(cat_name, "r") as f:
    #     for line in f:
    #         if line.startswith("#"):
    #             # SExtractor headers look like:
    #             #  #   1 NUMBER
    #             #  #   2 X_IMAGE
    #             parts = line.strip().split()
    #             if len(parts) >= 3:
    #                 colnames.append(parts[2])  # take third token = column name
    #         else:
    #             break  # stop when actual data starts

    # # Load the catalog into a pandas DataFrame
    # cat_df = pd.read_csv(
    #     cat_name,
    #     delim_whitespace=True,  # SExtractor catalogs are whitespace-delimited
    #     comment='#',             # Ignore header comment lines
    #     names=colnames
    # )

    # # Add a column of kron radius in pix from A
    # cat_df["KRON_A_PIX"] = cat_df["KRON_RADIUS"]*cat_df["A_IMAGE"] 

    # return seg_fits, cat_df

    return seg_fits

#------------------------------------------------------------------------------------------------------------------------

def sersic_to_image(brightness,
                    sma_values,
                    el_list,
                    pa_list,
                    shape,
                    center,
                    fill_value=np.nan,
                    oversample_factor=5):
    """
    Generate a 2D image from 1D list of Sersic profile values.

    Parameters:
    -----------
    brightness : 1D array 
        Array of intensity values.
    sma_values : list
        List of semi-major axis values (pix) where brightness was measured.
    el_list : 1D array 
        Array of ellipticities (same length as brightness).
    pa_list : 1D array 
        Array of position angles in radians (same length).
    shape : tuple (ny, nx)
        Output image shape.
    center : tuple (y0, x0)
        Center pixel.
    fill_value : value for unused pixels. Default np.nan

    Returns:
    --------
    img : 2D numpy array
        Interpolated 2D Sersic array
    """

    # Convert pa radians to degrees
    pa_list = np.rad2deg(pa_list)

    # Apply gaussian smoothing to el and pa lists
    el_list = gaussian_filter1d(el_list, sigma=2)
    pa_list = gaussian_filter1d(pa_list, sigma=2)
    
    # Separate out shape and center tuples
    ny, nx = shape
    y0, x0 = center

    # Coordinate grid
    y, x = np.indices((ny, nx))
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)  # circular radius as a proxy

    # Define a radius scale matching brightness/ellipticity/PA profiles
    # r_sample = np.linspace(0, r.max(), len(brightness))


    r_sample = sma_values


    # # Interpolate to finer radial scale
    # r_sample = np.linspace(sma_values[0], sma_values[-1],
    #                              oversample_factor * len(sma_values))

    # # Use cubic or linear interpolation depending on smoothness needs
    # brightness = interp1d(sma_values, brightness, kind='cubic')(r_sample)
    # el_list        = interp1d(sma_values, el_list,    kind='cubic')(r_sample)
    # pa_list         = interp1d(sma_values, pa_list,    kind='cubic')(r_sample)


    # Interpolators for ellipticity and PA
    e_interp = interp1d(r_sample, el_list, kind='linear',
                        bounds_error=False, fill_value=(el_list[0], el_list[-1]))
    pa_interp = interp1d(r_sample, pa_list, kind='linear',
                         bounds_error=False, fill_value=(pa_list[0], pa_list[-1]))

    # Evaluate interpolated ellipticity and PA at each pixel
    e_at_r = e_interp(r)
    pa_rad = np.deg2rad(pa_interp(r))  # convert back to radians

    # Compute a/b from ellipticity: a/b = 1 / (1 - e)
    a_over_b = 1.0 / (1.0 - e_at_r)

    # Rotate each pixel's coordinates by local PA
    x_rot =  dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)

    # Elliptical radius at each pixel
    r_ellip = np.sqrt(x_rot**2 + (y_rot * a_over_b)**2)

    # # Sort pixels by elliptical radius
    # flat_r_ellip = r_ellip.ravel()
    # flat_order = np.argsort(flat_r_ellip)

    # flat_img = np.full(flat_r_ellip.size, fill_value, dtype=float)

    # # Paint brightness values onto pixels sorted by elliptical radius
    # n_brightness = len(brightness)
    # flat_img[flat_order[:n_brightness]] = brightness[:min(n_brightness, flat_r_ellip.size)]

    # return flat_img.reshape((ny, nx))


    # Create interpolator for brightness
    brightness_interp = interp1d(r_sample, brightness, kind='linear',
                                 bounds_error=False, fill_value=(brightness[0], brightness[-1]))
    
    # Interpolate brightness at every pixel
    img = brightness_interp(r_ellip)
    # img = gaussian_filter(img, sigma=1.0)
    
    return img

#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

#------------------------------------------------------------------------------------------------------------------------
# NED/SIMBAD query on cluster
#------------------------------------------------------------------------------------------------------------------------

cln_ned_table = Ned.query_object(cln)

# ned_RA = cln_ned_table['RA'][0]
# ned_DEC = cln_ned_table['DEC'][0]
ned_z = cln_ned_table['Redshift'][0]

# For Simbad, 250 arcsec region query (about 951 pix)
simbad = Simbad()
simbad.add_votable_fields("otype","rvz_redshift")
cln_simbad_table = simbad.query_region(cln, radius=250*u.arcsec) 

objtypes = ['BiC','LSB','QSO','rG','BLL','EmG'] #types to search for

# Sort the possible BCG candidates by order of objtype above
order_map = {t: i for i, t in enumerate(objtypes)}
bcg_simbad_table = cln_simbad_table[np.isin(cln_simbad_table['otype'], objtypes)]
sorted_bcg_simbad_table = bcg_simbad_table[np.argsort([order_map[o] for o in bcg_simbad_table['otype']])]

# Get the coords of the first entry
simbad_RA = sorted_bcg_simbad_table['ra'][0]
simbad_DEC = sorted_bcg_simbad_table['dec'][0]

#------------------------------------------------------------------------------------------------------------------------
# Fetching coadds and running SExtractor
#------------------------------------------------------------------------------------------------------------------------


# Coadds and their headers (WCS)

# coadd_data_dict = {band: fits.open(coadd_fits_dict[band])[0].data for band in band_list}
# coadd_header_dict = {band: fits.open(coadd_fits_dict[band])[0].header for band in band_list}

coadd_g_data = fits.open(coadd_g_fits)[0].data; coadd_g_header = fits.open(coadd_g_fits)[0].header
coadd_r_data = fits.open(coadd_r_fits)[0].data; coadd_r_header = fits.open(coadd_r_fits)[0].header
coadd_i_data = fits.open(coadd_i_fits)[0].data; coadd_i_header = fits.open(coadd_i_fits)[0].header
coadd_z_data = fits.open(coadd_z_fits)[0].data; coadd_z_header = fits.open(coadd_z_fits)[0].header

# Run sextractor
print("------------------------------------")
print("Running initial SExtractor...")

mask_g_fits = run_sextractor(coadd_g_fits, band='g', run_num=0)
mask_r_fits = run_sextractor(coadd_r_fits, band='r', run_num=0)
mask_i_fits = run_sextractor(coadd_i_fits, band='i', run_num=0)
mask_z_fits = run_sextractor(coadd_z_fits, band='z', run_num=0)


# Load mask data
mask_g_data = mask_g_fits[0].data
mask_r_data = mask_r_fits[0].data
mask_i_data = mask_i_fits[0].data
mask_z_data = mask_z_fits[0].data


# Getting coadds photometric errors
print("------------------------------------")
print("Calculating photometric input errors...")

coadd_input_errors_mag = np.array(get_photometric_errors()) # Errors in units of mag

# print(coadd_input_errors_mag)

input_err_fractional_arr = (np.log(10)/(2.5))*coadd_input_errors_mag #sigma flux/flux.

# Generate input error array for each band
g_input_err_array = np.abs(coadd_g_data*input_err_fractional_arr[0]) #Units of counts/pix
r_input_err_array = np.abs(coadd_r_data*input_err_fractional_arr[1])
i_input_err_array = np.abs(coadd_i_data*input_err_fractional_arr[2])
z_input_err_array = np.abs(coadd_z_data*input_err_fractional_arr[3])

input_err_array = {'g': g_input_err_array, 'r': r_input_err_array, 'i': i_input_err_array, 'z': z_input_err_array}


# Display a coadd or mask
plt.figure()
plt.imshow(display(coadd_g_data)*255) # Data multiplied by 255 after stretching. Not sure why.
plt.gca().invert_yaxis() # Invert y since the image is "upside down"
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('g-band Coadd')
plt.savefig(f'{output_folder}/{cln}_g_coadd.png', dpi=1000, bbox_inches='tight')
plt.close()


#------------------------------------------------------------------------------------------------------------------------
# Obtaining/refining BCG coordinates
#------------------------------------------------------------------------------------------------------------------------

print("------------------------------------")
print("Getting BCG coordinates...")

# WCS (World Coordinate System) is same for all griz bands, so just using r 
r_wcs = WCS(coadd_r_header)


# Check if cluster is available in BCG coord csv. If not, use NED to obtain coords

if cln in bcg_coords_df['Name'].values:

    print(f"{cln} found in the BCG dataframe.")

    cluster_bcg_coords = bcg_coords_df[bcg_coords_df['Name'] == cln]

    cluster_bcg_RA = cluster_bcg_coords['RA'].iloc[0]
    cluster_bcg_DEC = cluster_bcg_coords['DEC'].iloc[0]

    #Initial coordinates. Floats
    x_bcg, y_bcg = r_wcs.world_to_pixel_values(cluster_bcg_RA, cluster_bcg_DEC)

    # Refined coordinates. Integers. Using r coadd only.
    x_bcg_ref, y_bcg_ref = recenter_bcg_coords(coadd_r_data, x_bcg, y_bcg, radius=25)

# else:

#     print(f"{cln} not found in BCG dataframe. Using NED coordinates.")
#     x_bcg, y_bcg = r_wcs.world_to_pixel_values(ned_RA, ned_DEC)

#     #Using larger search radius
#     x_bcg_ref, y_bcg_ref = recenter_bcg_coords(coadd_r_data, x_bcg, y_bcg, radius=100)


else:

    print(f"{cln} not found in BCG dataframe. Using SIMBAD coordinates.")
    x_bcg, y_bcg = r_wcs.world_to_pixel_values(simbad_RA, simbad_DEC)

    #Using slightly larger search radius
    x_bcg_ref, y_bcg_ref = recenter_bcg_coords(coadd_r_data, x_bcg, y_bcg, radius=40)


# Display center/refined center
plt.figure()
plt.imshow(display(coadd_r_data)*255)
plt.plot(x_bcg,y_bcg, 'ro', mfc = 'none', label='original')
plt.plot(x_bcg_ref,y_bcg_ref, 'go', mfc = 'none', label='refined')
plt.gca().invert_yaxis()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.xlim(x_bcg_ref - 300, x_bcg_ref + 300)
plt.ylim(y_bcg_ref - 300, y_bcg_ref + 300)
plt.title(f'{cln} r-band coadd BCG')
plt.legend()
plt.savefig(f'{output_folder}/{cln}_r_coadd_BCG_loc.png', dpi=1000, bbox_inches='tight')
plt.close()



#------------------------------------------------------------------------------------------------------------------------
# Calculating range of interest
#------------------------------------------------------------------------------------------------------------------------

# print("------------------------------------")
# print(f"Finding characteristic range for {cln}...")

d_A = cosmo.angular_diameter_distance(ned_z)  # Angular diameter distance (Mpc)

# Physical scale: how many kpc per arcsecond
scale = (d_A.to(u.kpc)*u.radian).value/rad_to_arcsec  # kpc/rad to kpc/arcsecond
# print(f"Scale: {scale:.2f} kpc/arcsec")

size_arcsec = 1.0*arcsec_per_pix_decam  # arcsec
size_kpc = size_arcsec*scale  # kpc

# print(f"1 pixel = {size_kpc:.2f} kpc")


# A85-based bounds
# lower_bound_kpc = 11 # Maybe placeholder value for now. Works for A85?
# lower_bound_pix = np.round(lower_bound_kpc/scale/arcsec_per_pix_decam)
# lower_bound_arcsec = np.round(lower_bound_kpc/scale)

# upper_bound_kpc = 75 # Maybe placeholder value for now. Works for A85?
# upper_bound_pix = np.round(upper_bound_kpc/scale/arcsec_per_pix_decam)
# upper_bound_arcsec = np.round(upper_bound_kpc/scale)

# print(f"A85-based lower bound of {lower_bound_kpc} kpc is {lower_bound_pix} pix or {lower_bound_arcsec} arcsec at z = {ned_z}")
# print(f"A85-based Upper bound of {upper_bound_kpc} kpc is {upper_bound_pix} pix or {upper_bound_arcsec} arcsec at z = {ned_z}")

# # Display donut around BCG to represent the range of interest from A85-bounds
# plt.figure()
# plt.imshow(display(coadd_r_data)*255)
# ax = plt.gca()
# circle1 = Circle((x_bcg_ref, y_bcg_ref), radius=lower_bound_pix, edgecolor='red', facecolor='none', lw=1) #A85 lower
# circle2 = Circle((x_bcg_ref, y_bcg_ref), radius=upper_bound_pix, edgecolor='red', facecolor='none', lw=1) #A85 upper
# # circle3 = Circle((x_bcg_ref, y_bcg_ref), radius=rkron_r_pix, edgecolor='blue', facecolor='none', lw=1) #kron radius
# ax.add_patch(circle1)
# ax.add_patch(circle2)
# # ax.add_patch(circle3)
# plt.gca().invert_yaxis()
# plt.xlabel('x (pixels)')
# plt.ylabel('y (pixels)')
# plt.xlim(x_bcg_ref - 500, x_bcg_ref + 500)
# plt.ylim(y_bcg_ref - 500, y_bcg_ref + 500)
# plt.title(f'{cln} r-Band Coadd Range of Interest')
# plt.tight_layout()
# plt.savefig(f'{output_folder}/{cln}_r_coadd_A85_range.png', dpi=1000, bbox_inches='tight')
# plt.close()




# Using rkron of the BCG from sextractor
# rkron_g_pix = sex_cat_g.loc[sex_cat_g['NUMBER'] == mask_g_data[int(y_bcg_ref), int(x_bcg_ref)]]["KRON_A_PIX"].values[0]
# rkron_g_kpc = rkron_g_pix*scale*arcsec_per_pix_decam
# rkron_g_arcsec = rkron_g_kpc/scale

# rkron_r_pix = sex_cat_r.loc[sex_cat_r['NUMBER'] == mask_r_data[int(y_bcg_ref), int(x_bcg_ref)]]["KRON_A_PIX"].values[0]
# rkron_r_kpc = rkron_r_pix*scale*arcsec_per_pix_decam
# rkron_r_arcsec = rkron_r_kpc/scale

# rkron_i_pix = sex_cat_i.loc[sex_cat_i['NUMBER'] == mask_i_data[int(y_bcg_ref), int(x_bcg_ref)]]["KRON_A_PIX"].values[0]
# rkron_i_kpc = rkron_i_pix*scale*arcsec_per_pix_decam
# rkron_i_arcsec = rkron_i_kpc/scale

# print(f'In g band, rkron is {rkron_g_pix} pix or {rkron_g_kpc} kpc or {rkron_g_arcsec} arcsec')
# print(f'In r band, rkron is {rkron_r_pix} pix or {rkron_r_kpc} kpc or {rkron_r_arcsec} arcsec')
# print(f'In i band, rkron is {rkron_i_pix} pix or {rkron_i_kpc} kpc or {rkron_i_arcsec} arcsec')



print("------------------------------------")
print("Entering SExtractor loop...")

# How many passes do you want?
iterations = 2 

for iter in range(1, iterations + 1):
    print(">"*108)
    print(f"STARTING ITERATION {iter}...")






    #------------------------------------------------------------------------------------------------------------------------
    # Applying masks and unmasking BCG
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Applying masks...")


    # Apply masks to all sources except the BCG
    bcg_unmsk_g_data = mask_single_coadd(coadd_g_data, mask_g_data, mask_g_data[int(y_bcg_ref), int(x_bcg_ref)],
                                         band='g', header=coadd_g_header, save_full_mask_im=True, run_num=iter)
    bcg_unmsk_r_data = mask_single_coadd(coadd_r_data, mask_r_data, mask_r_data[int(y_bcg_ref), int(x_bcg_ref)],
                                         band='r', header=coadd_r_header, save_full_mask_im=True, run_num=iter)
    bcg_unmsk_i_data = mask_single_coadd(coadd_i_data, mask_i_data, mask_i_data[int(y_bcg_ref), int(x_bcg_ref)],
                                         band='i', header=coadd_i_header, save_full_mask_im=True, run_num=iter)
    bcg_unmsk_z_data = mask_single_coadd(coadd_z_data, mask_z_data, mask_z_data[int(y_bcg_ref), int(x_bcg_ref)],
                                         band='z', header=coadd_z_header, save_full_mask_im=True, run_num=iter)
    
    bcg_unmsk_data_dict = {'g': bcg_unmsk_g_data, 'r': bcg_unmsk_r_data, 'i': bcg_unmsk_i_data, 'z': bcg_unmsk_z_data}

    lsp_mask_g_data = fits.open(lsp_mask_g_fits)[0].data
    lsp_mask_r_data = fits.open(lsp_mask_r_fits)[0].data
    lsp_mask_i_data = fits.open(lsp_mask_i_fits)[0].data
    lsp_mask_z_data = fits.open(lsp_mask_z_fits)[0].data

    region_unmsk_g_data = apply_lsp_mask(coadd_g_data, lsp_mask_g_data, int(y_bcg_ref), int(x_bcg_ref), band='g',
                                          header=coadd_g_header, save_full_mask_im=True, unmsk_scale=100, run_num=iter)
    region_unmsk_r_data = apply_lsp_mask(coadd_r_data, lsp_mask_r_data, int(y_bcg_ref), int(x_bcg_ref), band='r',
                                          header=coadd_r_header, save_full_mask_im=True, unmsk_scale=100, run_num=iter)
    region_unmsk_i_data = apply_lsp_mask(coadd_i_data, lsp_mask_i_data, int(y_bcg_ref), int(x_bcg_ref), band='i',
                                          header=coadd_i_header, save_full_mask_im=True, unmsk_scale=100, run_num=iter)
    region_unmsk_z_data = apply_lsp_mask(coadd_z_data, lsp_mask_z_data, int(y_bcg_ref), int(x_bcg_ref), band='z',
                                          header=coadd_z_header, save_full_mask_im=True, unmsk_scale=100, run_num=iter)
    
    region_unmsk_data_dict = {'g': region_unmsk_g_data, 'r': region_unmsk_r_data, 'i': region_unmsk_i_data, 'z': region_unmsk_z_data}

    cutout_mask_g = run_sextractor(f'{output_folder}/coadd_region_cutout_g_iter_{iter}.fits', band='g',run_num=iter, detect_thresh_param='2.0', tag='lsp')
    cutout_mask_r = run_sextractor(f'{output_folder}/coadd_region_cutout_r_iter_{iter}.fits', band='r',run_num=iter, detect_thresh_param='2.0', tag='lsp')
    cutout_mask_i = run_sextractor(f'{output_folder}/coadd_region_cutout_i_iter_{iter}.fits', band='i',run_num=iter, detect_thresh_param='2.0', tag='lsp')
    cutout_mask_z = run_sextractor(f'{output_folder}/coadd_region_cutout_z_iter_{iter}.fits', band='z',run_num=iter, detect_thresh_param='2.0', tag='lsp')


    for band in band_list:
        plt.figure()
        cmap = plt.cm.viridis.copy() 
        # cmap.set_bad(color='black')#Setting masked pixels to black
        cmap.set_under(cmap(0))
        plt.imshow((display(bcg_unmsk_data_dict[band]).filled(fill_value=-1))*255, cmap=cmap, vmin=0)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        plt.xlim(x_bcg_ref - 500, x_bcg_ref + 500)
        plt.ylim(y_bcg_ref - 500, y_bcg_ref + 500)
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.title(f'{cln} Masked {band}-band Coadd')
        plt.savefig(f'{output_folder}/{cln}_{band}_coadd_masked_iter_{iter}.png', dpi=1000, bbox_inches='tight')
        plt.close()

    for band in band_list:
        plt.figure()
        cmap = plt.cm.viridis.copy() 
        # cmap.set_bad(color='black')#Setting masked pixels to black
        cmap.set_under(cmap(0))
        plt.imshow((display(region_unmsk_data_dict[band]).filled(fill_value=-1))*255, cmap=cmap, vmin=0)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        plt.xlim(x_bcg_ref - 1000, x_bcg_ref + 1000)
        plt.ylim(y_bcg_ref - 1000, y_bcg_ref + 1000)
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.title(f'{cln} Masked {band}-band Coadd')
        plt.savefig(f'{output_folder}/{cln}_{band}_coadd_region_unmasked_iter_{iter}.png', dpi=1000, bbox_inches='tight')
        plt.close()


    #------------------------------------------------------------------------------------------------------------------------
    # Measuring background signal
    #------------------------------------------------------------------------------------------------------------------------

    # print("------------------------------------")
    # print("Measuring backgound signal...")


    # Calculating largest ellipse to fit in coadd image
    ny, nx = coadd_g_data.shape #Assuming using same size patches for griz

    # Distance from ellipse center to edges
    dist_left = int(x_bcg_ref); dist_right = nx - int(x_bcg_ref) 
    dist_top = int(y_bcg_ref); dist_bottom = ny - int(y_bcg_ref)

    # Find max sma
    max_sma = 0.98*min(dist_left, dist_right, dist_top, dist_bottom)


    # Spacings to fit annuli. lin_spacings below is default of fit_ellip_annuli()
    # log_spacings = np.logspace(1,3.2,15) 
    # lin_spacings = np.linspace(10, 1900, 30) #Original linspace used

    # lin_spacings, spacing_step = np.linspace(10, 2900, num=40, retstep=True) #This works for manual maxsma
    # lin_spacings, spacing_step = np.linspace(10, max_sma, num=40, retstep=True) #This works for manual maxsma



    # # Fitting elliptical annuli. Getting intensity sums, sum errors, and pix areas.
    # sums_g, sums_err_g, areas_g = fit_ellip_annuli('g', bcg_unmsk_g_data, 
    #                                 np.array([int(x_bcg_ref), int(y_bcg_ref)]), 
    #                                 g_input_err_array, spacings=lin_spacings, theta=0, run_num=iter)
    # sums_r, sums_err_r, areas_r = fit_ellip_annuli('r', bcg_unmsk_r_data, 
    #                                 np.array([int(x_bcg_ref), int(y_bcg_ref)]), 
    #                                 r_input_err_array, spacings=lin_spacings, theta=0, run_num=iter)
    # sums_i, sums_err_i, areas_i = fit_ellip_annuli('i', bcg_unmsk_i_data, 
    #                                 np.array([int(x_bcg_ref), int(y_bcg_ref)]), 
    #                                 i_input_err_array, spacings=lin_spacings, theta=0, run_num=iter)

    # # Mean intensities (counts/pix)
    # means_g_bkg = np.array([a/b for a,b in zip(sums_g,areas_g)])
    # means_r_bkg = np.array([a/b for a,b in zip(sums_r,areas_r)])
    # means_i_bkg = np.array([a/b for a,b in zip(sums_i,areas_i)])

    # # Errors in mean intensities
    # means_g_bkg_err = np.array([a/b for a,b in zip(sums_err_g,areas_g)])
    # means_r_bkg_err = np.array([a/b for a,b in zip(sums_err_r,areas_r)])
    # means_i_bkg_err = np.array([a/b for a,b in zip(sums_err_i,areas_i)])

    # # Finding mean intensity after slope levels off

    # bkg_grad_threshold = 0.01

    # # Calculating gradient. Also specifying step size
    # means_g_bkg_grad = np.gradient(means_g_bkg, spacing_step)
    # means_r_bkg_grad = np.gradient(means_r_bkg, spacing_step)
    # means_i_bkg_grad = np.gradient(means_i_bkg, spacing_step)

    # # Gradient error propagation
    # means_g_bkg_grad_err = np.sqrt(np.roll(means_g_bkg_err, -1)**2 + np.roll(means_g_bkg_err, 1)**2)/(2*spacing_step)
    # means_r_bkg_grad_err = np.sqrt(np.roll(means_r_bkg_err, -1)**2 + np.roll(means_r_bkg_err, 1)**2)/(2*spacing_step)
    # means_i_bkg_grad_err = np.sqrt(np.roll(means_i_bkg_err, -1)**2 + np.roll(means_i_bkg_err, 1)**2)/(2*spacing_step)

    # # Finding value fitting threshold criteria
    # bkg_slope_mask_g = np.abs(means_g_bkg_grad) < bkg_grad_threshold
    # bkg_slope_mask_r = np.abs(means_r_bkg_grad) < bkg_grad_threshold
    # bkg_slope_mask_i = np.abs(means_i_bkg_grad) < bkg_grad_threshold

    # bkg_mean_val_g = np.nanmean(means_g_bkg[bkg_slope_mask_g])
    # bkg_mean_val_r = np.nanmean(means_r_bkg[bkg_slope_mask_r])
    # bkg_mean_val_i = np.nanmean(means_i_bkg[bkg_slope_mask_i])

    # # Masked gradient error
    # means_g_bkg_grad_err_masked = means_g_bkg_grad_err[bkg_slope_mask_g]
    # means_r_bkg_grad_err_masked = means_r_bkg_grad_err[bkg_slope_mask_r]
    # means_i_bkg_grad_err_masked = means_i_bkg_grad_err[bkg_slope_mask_i]

    # # Unweighted-mean error prop
    # bkg_mean_err_g = np.sqrt(np.sum(means_g_bkg_grad_err_masked**2))/len(means_g_bkg[bkg_slope_mask_g])
    # bkg_mean_err_r = np.sqrt(np.sum(means_r_bkg_grad_err_masked**2))/len(means_r_bkg[bkg_slope_mask_r])
    # bkg_mean_err_i = np.sqrt(np.sum(means_i_bkg_grad_err_masked**2))/len(means_i_bkg[bkg_slope_mask_i])


    # print(f"g/r/i background mean values: {bkg_mean_val_g}/{bkg_mean_val_r}/{bkg_mean_val_i} with threshold gradient of {bkg_grad_threshold} +-  {bkg_mean_err_g}/{bkg_mean_err_r}/{bkg_mean_err_i}")



    # # Plotting background intensities (log plot)
    # plt.figure(figsize=(6,4))
    # # plt.scatter(lin_spacings[1:], means_g_bkg, s = 50, color= 'g', marker ='o', label='g')
    # # plt.scatter(lin_spacings[1:], means_r_bkg, s = 50, color= 'r', marker ='o', label='r')
    # # plt.scatter(lin_spacings[1:], means_i_bkg, s = 50, color= 'k', marker ='o', label='i')

    # plt.errorbar(lin_spacings[1:], means_g_bkg, yerr=means_g_bkg_err,
    #              fmt='o', color=color_dict['g'], label='g', capsize=3, markersize=3)
    # plt.errorbar(lin_spacings[1:], means_r_bkg, yerr=means_r_bkg_err,
    #              fmt='o', color=color_dict['r'], label='r', capsize=3, markersize=3)
    # plt.errorbar(lin_spacings[1:], means_i_bkg, yerr=means_i_bkg_err,
    #              fmt='o', color=color_dict['i'], label='i', capsize=3, markersize=3)
    # plt.plot(lin_spacings[1:], means_g_bkg, color_dict['g'])
    # plt.plot(lin_spacings[1:], means_r_bkg, color_dict['r'])
    # plt.plot(lin_spacings[1:], means_i_bkg, color_dict['i'])

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.xlabel('SMA (pix)')
    # plt.ylabel('Mean Intensity (counts/pix)')
    # plt.title(f'{cln} Background Intensities')
    # plt.savefig(f'{output_folder}/{cln}_background_signal_ylog_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    # plt.close()


    

    # # Plotting background intensities (linear plot)
    # plt.figure(figsize=(6,4))
    # # plt.scatter(lin_spacings[1:], means_g_bkg, s = 50, color= 'g', marker ='o', label='g')
    # # plt.scatter(lin_spacings[1:], means_r_bkg, s = 50, color= 'r', marker ='o', label='r')
    # # plt.scatter(lin_spacings[1:], means_i_bkg, s = 50, color= 'k', marker ='o', label='i')

    # plt.errorbar(lin_spacings[1:], means_g_bkg, yerr=means_g_bkg_err,
    #              fmt='o', color=color_dict['g'], label='g', capsize=3, markersize=3)
    # plt.errorbar(lin_spacings[1:], means_r_bkg, yerr=means_r_bkg_err,
    #              fmt='o', color=color_dict['r'], label='r', capsize=3, markersize=3)
    # plt.errorbar(lin_spacings[1:], means_i_bkg, yerr=means_i_bkg_err,
    #              fmt='o', color=color_dict['i'], label='i', capsize=3, markersize=3)
    # plt.plot(lin_spacings[1:], means_g_bkg, color_dict['g'])
    # plt.plot(lin_spacings[1:], means_r_bkg, color_dict['r'])
    # plt.plot(lin_spacings[1:], means_i_bkg, color_dict['i'])

    # plt.axhline(y=0, color='k', linestyle='--')

    # plt.xscale('log')

    # plt.ylim(-1e-1, 1e-1)

    # plt.legend()
    # plt.xlabel('SMA (pix)')
    # plt.ylabel('Mean Intensity (counts/pix)')
    # plt.title(f'{cln} Background Intensities')
    # plt.savefig(f'{output_folder}/{cln}_background_signal_lin_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    # plt.close()



    #------------------------------------------------------------------------------------------------------------------------
    # Fitting Isophotes 
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Fitting isophotes...")

    isophotes_g = isophote_fitter(bcg_unmsk_g_data, x0=int(x_bcg_ref), y0=int(y_bcg_ref), max_SMA=max_sma)
    isophotes_r = isophote_fitter(bcg_unmsk_r_data, x0=int(x_bcg_ref), y0=int(y_bcg_ref), max_SMA=max_sma)
    isophotes_i = isophote_fitter(bcg_unmsk_i_data, x0=int(x_bcg_ref), y0=int(y_bcg_ref), max_SMA=max_sma)
    isophotes_z = isophote_fitter(bcg_unmsk_z_data, x0=int(x_bcg_ref), y0=int(y_bcg_ref), max_SMA=max_sma)

    # Display isophotes

    coadd_data_dict = {'g': coadd_g_data, 'r': coadd_r_data, 'i': coadd_i_data, 'z': coadd_z_data}
    coadd_masked_data_dict = {'g': bcg_unmsk_g_data, 'r': bcg_unmsk_r_data, 'i': bcg_unmsk_i_data, 'z': bcg_unmsk_z_data}
    isophotes_dict = {'g': isophotes_g, 'r': isophotes_r, 'i': isophotes_i, 'z': isophotes_z}

    el_pixels = [] #List to store pixel indices in each ellipse aperture

    for band in band_list:

        plt.figure(figsize=(10,8))
        plt.imshow(display(coadd_data_dict[band])*255)

        sma_lim = max(iso.sma for iso in isophotes_dict[band]) #Get largest sma
        x0 = isophotes_dict[band][0].x0 #This is just grabbing the center from an isophote to use in xlim, ylim
        y0 = isophotes_dict[band][0].y0

        # pix_idx = [] #Storing aperture pix indices here. Should be a list of length 3

        for iso in isophotes_dict[band]:
            e = mpl_Ellipse(xy=(iso.x0, iso.y0), width=2*iso.sma, height=2*iso.sma*(1-iso.eps), \
                        angle=iso.pa*180/np.pi, edgecolor=color_dict[band], facecolor='none')
            plt.gca().add_patch(e) # Plotting the ellipse

            # This is for determining pixel indices of annuli for error prop
        #     aperture = EllipticalAperture((iso.x0,iso.y0), a=iso.sma, b=iso.sma*(1-iso.eps), theta=iso.pa)
        #     mask = aperture.to_mask(method='exact').to_image((ny,nx)) 
        #     indices = np.argwhere(mask > 0.4) # Indices of pixels inside the ellipse
        #     pix_idx.append(indices)

        # el_pixels.append(pix_idx)
            
        # plt.plot(isophotes_g.x0[0],isophotes_g.y0[0], 'r+')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        # plt.xlim(isophotes_dict[band].x0[0] - 1100, isophotes_dict[band].x0[0] + 1100)
        # plt.ylim(isophotes_dict[band].y0[0] - 1100, isophotes_dict[band].y0[0] + 1100)
        pad = 0.05*sma_lim
        plt.xlim(x0 - sma_lim - pad, x0 + sma_lim + pad)
        plt.ylim(y0 - sma_lim - pad, y0 + sma_lim + pad)
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.title(f'{cln} {band}-band isophotes fixed center')
        plt.savefig(f'{output_folder}/{cln}_{band}_isophotes_iter_{iter}.png', dpi=1000, bbox_inches='tight')
        plt.close()

    # print(el_pixels)
    # print(len(el_pixels))

    # annuli_pix_idx = []

    # #For each band's set of aperture pix coordinates..
    # for band_pix in el_pixels:
    # # Create list to store annuli coords 
    #     band_annuli = []   
    #     # Loop through (number of isophotes - 1) times    
    #     for i in range(len(band_pix)-1):
    #         # Grab inner and out aperture pix coords
    #         inner = band_pix[i]
    #         outer = band_pix[i+1]

    #         # Convert rows to sets of tuples (for set operations)
    #         inner_set = set(map(tuple, inner))
    #         outer_set = set(map(tuple, outer))

    #         # Get annulus pixel coords
    #         annulus = np.array(list(outer_set - inner_set))
    #         # Append annulus pixel coords to list 
    #         band_annuli.append(annulus)

    #     # Append list of band's annulus pix coords to list
    #     annuli_pix_idx.append(band_annuli)


    # annuli_pix_sums = []
    # for band in annuli_pix_idx:
    #     band_sums = []
    #     for annulus_pix in band:
    #         y_coords = annulus_pix[:, 0]
    #         x_coords = annulus_pix[:, 1]

    #         annulus_sum = bcg_unmsk_g_data[y_coords, x_coords].sum()
    #         band_sums.append(annulus_sum)

    #         # print(annulus_sum)
    #     annuli_pix_sums.append(band_sums)
    # annuli_pix_sums = np.array(annuli_pix_sums)

    

    #------------------------------------------------------------------------------------------------------------------------
    # Surface brightness
    #------------------------------------------------------------------------------------------------------------------------
    print("------------------------------------")
    print("Plotting surface brightness contours...")

    for band in band_list:
        draw_lsb_contours(coadd_masked_data_dict[band], band=band)


    print("------------------------------------")
    print("Calculating surface brightness...")



    # Extract cumulative intensity sums for each isophote
    # So the sum in each annulus
    annular_sums_g = np.array([isophotes_g.tflux_e[i+1] - isophotes_g.tflux_e[i] \
                    for i in range(len(isophotes_g.tflux_e)-1)])
    annular_sums_r = np.array([isophotes_r.tflux_e[i+1] - isophotes_r.tflux_e[i] \
                    for i in range(len(isophotes_r.tflux_e)-1)])
    annular_sums_i = np.array([isophotes_i.tflux_e[i+1] - isophotes_i.tflux_e[i] \
                    for i in range(len(isophotes_i.tflux_e)-1)])
    annular_sums_z = np.array([isophotes_z.tflux_e[i+1] - isophotes_z.tflux_e[i] \
                    for i in range(len(isophotes_z.tflux_e)-1)])
    
    #Dictionary to store annuli fluxes for all bands
    annuli_fluxes = {} 
    annuli_idx = {}
    annuli_equiv_areas_pix = {}
    for band in band_list:
        band_annuli_fluxes = []
        band_annuli_idx = []
        band_annuli_area_pix = []
        for i in range(len(isophotes_dict[band])-1):
            annulus_flux, annulus_pix_idx = annulus_flux_measure(coadd_masked_data_dict[band], isophotes_dict[band][i], isophotes_dict[band][i+1])
            band_annuli_fluxes.append(annulus_flux)
            band_annuli_idx.append(annulus_pix_idx)
            band_annuli_area_pix.append(np.count_nonzero(annulus_pix_idx))
            # print(annulus_flux)
        annuli_fluxes[band] = np.array(band_annuli_fluxes)
        annuli_idx[band] = band_annuli_idx
        annuli_equiv_areas_pix[band] = np.array(band_annuli_area_pix)

    # print(annuli_fluxes['g'])
    # print(annuli_fluxes['r'])
    # print(annuli_fluxes['i'])

    # print(annular_sums_g)
    # print(annular_sums_r)
    # print(annular_sums_i)

    print(np.round(abs(((annular_sums_g - annuli_fluxes['g'])/annular_sums_g))*100, 2))
    print(np.round(abs(((annular_sums_r - annuli_fluxes['r'])/annular_sums_r))*100, 2))
    print(np.round(abs(((annular_sums_i - annuli_fluxes['i'])/annular_sums_i))*100, 2))
    print(np.round(abs(((annular_sums_z - annuli_fluxes['z'])/annular_sums_z))*100, 2))


    # Calculating pixel sum errors
    annuli_sum_errs = {}
    
    for band in band_list:
        errs = []
        for mask in annuli_idx[band]: 
            pix_err = input_err_array[band][mask]
            errs.append(np.sqrt(np.sum(pix_err**2)))
        annuli_sum_errs[band] = np.array(errs)

    # print(annuli_sum_errs)


    # Manually calculate the GEOMTRICAL annulus areas pi*a*b. 
    annular_areas_g = [((1 - isophotes_g.eps[i+1])*isophotes_g.sma[i+1]**2 \
                        - (1 - isophotes_g.eps[i])*isophotes_g.sma[i]**2) \
        for i in range(len(isophotes_g.sma)-1)]
    annular_areas_g = np.pi*np.array(annular_areas_g) 

    annular_areas_r = [((1 - isophotes_r.eps[i+1])*isophotes_r.sma[i+1]**2 \
                        - (1 - isophotes_r.eps[i])*isophotes_r.sma[i]**2) \
        for i in range(len(isophotes_r.sma)-1)]
    annular_areas_r = np.pi*np.array(annular_areas_r) 

    annular_areas_i = [((1 - isophotes_i.eps[i+1])*isophotes_i.sma[i+1]**2 \
                        - (1 - isophotes_i.eps[i])*isophotes_i.sma[i]**2) \
        for i in range(len(isophotes_i.sma)-1)]
    annular_areas_i = np.pi*np.array(annular_areas_i) 

    annular_areas_z = [((1 - isophotes_z.eps[i+1])*isophotes_z.sma[i+1]**2 \
                        - (1 - isophotes_z.eps[i])*isophotes_z.sma[i]**2) \
        for i in range(len(isophotes_z.sma)-1)]
    annular_areas_z = np.pi*np.array(annular_areas_z) 

    annular_areas_dict = {'g': annular_areas_g, 'r': annular_areas_r, 'i': annular_areas_i, 'z': annular_areas_z}


    # Dividing annulus intensity sums by areas to get annular intensity means
    # annular_means_g = [a/b for a,b in zip(annular_sums_g,annular_areas_g)] OLD METHOD WITH GEOMETRIC AREAS AND ISO SUBTRACTION FLUX
    # annular_means_r = [a/b for a,b in zip(annular_sums_r,annular_areas_r)]
    # annular_means_i = [a/b for a,b in zip(annular_sums_i,annular_areas_i)]
    # annular_means_z = [a/b for a,b in zip(annular_sums_z,annular_areas_z)]
    annular_means_g = [a/b for a,b in zip(annuli_fluxes['g'],annuli_equiv_areas_pix['g'])]
    annular_means_r = [a/b for a,b in zip(annuli_fluxes['r'],annuli_equiv_areas_pix['r'])]
    annular_means_i = [a/b for a,b in zip(annuli_fluxes['i'],annuli_equiv_areas_pix['i'])]
    annular_means_z = [a/b for a,b in zip(annuli_fluxes['z'],annuli_equiv_areas_pix['z'])]




    # print(annular_areas_g)
    # print(annuli_equiv_areas_pix['g'])
    # print(isophotes_g.sma)
    # print(isophotes_g.eps)
    # print(stopscript)


    # annular mean errors (pix)

    annuli_mean_error_pix = {}

    for band in band_list:
        annuli_mean_error_pix[band] = np.abs(annuli_sum_errs[band]/annuli_equiv_areas_pix[band])


    # print(annuli_mean_error_pix)

    # sma_values (x-values) are same for all gri bands. Just using r band 
    # Excluding final value to match size of other lists??
    sma_vals_pix = np.array([iso.sma for iso in isophotes_dict['r'][:-1]])

    # Converts SMA from pix to arcsec.
    sma_vals_arcsec = arcsec_per_pix_decam*np.array(sma_vals_pix)

    # Get SMA vals in kpc too
    sma_vals_kpc = scale*np.array(sma_vals_arcsec)

    # Getting indices from upper_bound_pix
    # bkg_sample_indices = np.where(lin_spacings > upper_bound_pix)[0] #### MUST ADDRESS THIS

    # Remove last array index to align with means_BAND_bkg length
    # bkg_sample_indices = bkg_sample_indices[:-1] 

    # Subtracting background values counts from annular means.

    # Uncertainty in background from variance in the last 20 points. Quadrature addition. sqrt(avg value per pixel). Uncertainty in mean is divided 
    # again by sqrt(areas)

    # print(f"g band background avg OLD METHOD: {np.mean([means_g_bkg[i] for i in bkg_sample_indices])}")
    # print(f"r band background avg OLD METHOD: {np.mean([means_r_bkg[i] for i in bkg_sample_indices])}")
    # print(f"i band background avg OLD METHOD: {np.mean([means_i_bkg[i] for i in bkg_sample_indices])}")

    # Subtracting off background signal
    # annular_means_g = annular_means_g - np.mean([means_g_bkg[i] for i in bkg_sample_indices]) #Old method
    # annular_means_r = annular_means_r - np.mean([means_r_bkg[i] for i in bkg_sample_indices])
    # annular_means_i = annular_means_i - np.mean([means_i_bkg[i] for i in bkg_sample_indices])


    # Saving intensity data to csv
    int_prof_df = pd.DataFrame({
        'sma_pix': sma_vals_pix,
        'sma_arcsec': sma_vals_arcsec,
        'sma_kpc': sma_vals_kpc,
        'mean_int_g': annular_means_g, #In units of counts/pix
        'mean_int_g_err': annuli_mean_error_pix['g'],
        'mean_int_r': annular_means_r,
        'mean_int_r_err': annuli_mean_error_pix['r'],
        'mean_int_i': annular_means_i,
        'mean_int_i_err': annuli_mean_error_pix['i'],
        'mean_int_z': annular_means_z,
        'mean_int_z_err': annuli_mean_error_pix['z'],
    })





 
    fig, ax = plt.subplots(figsize=(8, 6))

    # Primary plot (bottom x-axis pix)
    # ax.semilogx(sma_vals_pix, annular_means_g, color=color_dict['g'], marker='o', label='g')
    # ax.semilogx(sma_vals_pix, annular_means_r, color=color_dict['r'], marker='o', label='r')
    # ax.semilogx(sma_vals_pix, annular_means_i, color=color_dict['i'], marker='o', label='i')

    ax.errorbar(sma_vals_pix, annular_means_g, yerr=annuli_mean_error_pix['g'], color=color_dict['g'], fmt='o-', capsize=6, markersize=6, label='g')
    ax.errorbar(sma_vals_pix, annular_means_r, yerr=annuli_mean_error_pix['r'], color=color_dict['r'], fmt='o-', capsize=6, markersize=6, label='r')
    ax.errorbar(sma_vals_pix, annular_means_i, yerr=annuli_mean_error_pix['i'], color=color_dict['i'], fmt='o-', capsize=6, markersize=6, label='i')
    ax.errorbar(sma_vals_pix, annular_means_z, yerr=annuli_mean_error_pix['z'], color=color_dict['z'], fmt='o-', capsize=6, markersize=6, label='z')

    ax.set_xscale('log')

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    ax.set_ylim(-1e-1,1e-1) 

    ax.set_xlabel('SMA (pix)')
    ax.set_ylabel('Annulus Mean Intensity (counts/pix)')
    ax.set_title(f'{cln} Isophote Annuli Mean Intensities (No background sub)')

    # Top x-axis using arcsec values
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_arcsec[0], sma_vals_arcsec[-1])
    ax_top.set_xlabel('SMA (arcsec)')

    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_isophote_intensities_nobacksub_lin_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()




    # Subtracting background values (attempting different methods)
    # annular_means_g = annular_means_g - bkg_mean_val_g #Gradient threshold method. ORIGINAL BACKGROUND METHOD
    # annular_means_r = annular_means_r - bkg_mean_val_r
    # annular_means_i = annular_means_i - bkg_mean_val_i

    # print(f'minimum intensity: g/r/i/z {min(annular_means_g)}, {min(annular_means_r)}, {min(annular_means_i)}, {min(annular_means_z)} counts/pix')

    # annular_means_g = annular_means_g - min(annular_means_g) 
    # annular_means_r = annular_means_r - min(annular_means_r) 
    # annular_means_i = annular_means_i - min(annular_means_i) 

    # print(f'mean intensity of last 4 points: g/r/i/z {np.mean(annular_means_g[-4:])}, {np.mean(annular_means_r[-4:])}, {np.mean(annular_means_i[-4:])}, {np.mean(annular_means_z[-4:])} counts/pix')

    # annular_means_g = annular_means_g - np.mean(annular_means_g[-4:]) #Subtract mean of the last 5 points
    # annular_means_r = annular_means_r - np.mean(annular_means_r[-4:]) 
    # annular_means_i = annular_means_i - np.mean(annular_means_i[-4:]) 


    # CURRENT method of background subtraction: Find minimum value and average n_el points around that value
    n_el = 5
    half_n = n_el//2

    min_int_g_index = np.argmin(annular_means_g) #Get mean of n elements around min value
    min_int_r_index = np.argmin(annular_means_r)
    min_int_i_index = np.argmin(annular_means_i)
    min_int_z_index = np.argmin(annular_means_z)


    start_index_g = max(0, min_int_g_index - half_n)
    end_index_g = min(len(annular_means_g), min_int_g_index + half_n + 1)

    start_index_r = max(0, min_int_r_index - half_n)
    end_index_r = min(len(annular_means_r), min_int_r_index + half_n + 1)

    start_index_i = max(0, min_int_i_index - half_n)
    end_index_i = min(len(annular_means_i), min_int_i_index + half_n + 1)

    start_index_z = max(0, min_int_z_index - half_n)
    end_index_z = min(len(annular_means_z), min_int_z_index + half_n + 1)

    #Set the background values for the rest of the iterations
    if iter == 1: 

        #Regular mean method (all points treated equally)
        # bkg_val_g = np.mean(annular_means_g[start_index_g:end_index_g])
        # bkg_val_err_g = np.sqrt(np.sum((annuli_mean_error_pix['g'][start_index_g:end_index_g]**2)))/n_el
        # bkg_val_r = np.mean(annular_means_r[start_index_r:end_index_r])
        # bkg_val_err_r = np.sqrt(np.sum((annuli_mean_error_pix['r'][start_index_r:end_index_r]**2)))/n_el
        # bkg_val_i = np.mean(annular_means_i[start_index_i:end_index_i])
        # bkg_val_err_i = np.sqrt(np.sum((annuli_mean_error_pix['i'][start_index_i:end_index_i]**2)))/n_el
        # bkg_val_z = np.mean(annular_means_z[start_index_z:end_index_z])
        # bkg_val_err_z = np.sqrt(np.sum((annuli_mean_error_pix['z'][start_index_z:end_index_z]**2)))/n_el

        # Weighted mean method
        weights_g = 1.0/annuli_mean_error_pix['g'][start_index_g:end_index_g]**2
        bkg_val_g = np.sum(weights_g*annular_means_g[start_index_g:end_index_g])/np.sum(weights_g)
        bkg_val_err_g = 1.0/np.sqrt(np.sum(weights_g))

        weights_r = 1.0/annuli_mean_error_pix['r'][start_index_r:end_index_r]**2
        bkg_val_r = np.sum(weights_r*annular_means_r[start_index_r:end_index_r])/np.sum(weights_r)
        bkg_val_err_r = 1.0/np.sqrt(np.sum(weights_r))

        weights_i = 1.0/annuli_mean_error_pix['i'][start_index_i:end_index_i]**2
        bkg_val_i = np.sum(weights_i*annular_means_i[start_index_i:end_index_i])/np.sum(weights_i)
        bkg_val_err_i = 1.0/np.sqrt(np.sum(weights_i))

        weights_z = 1.0/annuli_mean_error_pix['z'][start_index_z:end_index_z]**2
        bkg_val_z = np.sum(weights_z*annular_means_z[start_index_z:end_index_z])/np.sum(weights_z)
        bkg_val_err_z = 1.0/np.sqrt(np.sum(weights_z))

        # Intrinsic scatter error
        scatter_err_g = np.std(annular_means_g[start_index_g:end_index_g], ddof=1)
        scatter_err_r = np.std(annular_means_r[start_index_r:end_index_r], ddof=1)
        scatter_err_i = np.std(annular_means_i[start_index_i:end_index_i], ddof=1)
        scatter_err_z = np.std(annular_means_z[start_index_z:end_index_z], ddof=1)

    annular_means_g = annular_means_g - bkg_val_g
    annular_means_r = annular_means_r - bkg_val_r
    annular_means_i = annular_means_i - bkg_val_i
    annular_means_z = annular_means_z - bkg_val_z

    annuli_mean_error_pix['g'] = np.sqrt(annuli_mean_error_pix['g']**2 + bkg_val_err_g**2 + scatter_err_g**2)
    annuli_mean_error_pix['r'] = np.sqrt(annuli_mean_error_pix['r']**2 + bkg_val_err_r**2 + scatter_err_r**2)
    annuli_mean_error_pix['i'] = np.sqrt(annuli_mean_error_pix['i']**2 + bkg_val_err_i**2 + scatter_err_i**2)
    annuli_mean_error_pix['z'] = np.sqrt(annuli_mean_error_pix['z']**2 + bkg_val_err_z**2 + scatter_err_z**2)



    print(f'mean intensity of {n_el} points around the min value: g/r/i/z {bkg_val_g}, {bkg_val_r}, {bkg_val_i}, {bkg_val_z} counts/pix')


    int_prof_df['mean_int_g_back_sub'] = annular_means_g #Adding background subtraction intensities
    int_prof_df['mean_int_r_back_sub'] = annular_means_r
    int_prof_df['mean_int_i_back_sub'] = annular_means_i
    int_prof_df['mean_int_z_back_sub'] = annular_means_z

    int_prof_df['mean_int_g_back_sub_err'] = annuli_mean_error_pix['g']
    int_prof_df['mean_int_r_back_sub_err'] = annuli_mean_error_pix['r']
    int_prof_df['mean_int_i_back_sub_err'] = annuli_mean_error_pix['i']
    int_prof_df['mean_int_z_back_sub_err'] = annuli_mean_error_pix['z']

    int_prof_df.to_csv(f'{output_folder}/{cln}_intensity_profile_iter_{iter}.csv', index=False)



    # Plotting mean intensities (counts/pix)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Primary plot (bottom x-axis pix)
    # ax.loglog(sma_vals_pix, annular_means_g, color=color_dict['g'], marker='o', label='g')
    # ax.loglog(sma_vals_pix, annular_means_r, color=color_dict['r'], marker='o', label='r')
    # ax.loglog(sma_vals_pix, annular_means_i, color=color_dict['i'], marker='o', label='i')

    ax.errorbar(sma_vals_pix, annular_means_g, yerr=annuli_mean_error_pix['g'], color=color_dict['g'], fmt='o-', capsize=6, markersize=6, label='g')
    ax.errorbar(sma_vals_pix, annular_means_r, yerr=annuli_mean_error_pix['r'], color=color_dict['r'], fmt='o-', capsize=6, markersize=6, label='r')
    ax.errorbar(sma_vals_pix, annular_means_i, yerr=annuli_mean_error_pix['i'], color=color_dict['i'], fmt='o-', capsize=6, markersize=6, label='i')
    ax.errorbar(sma_vals_pix, annular_means_z, yerr=annuli_mean_error_pix['z'], color=color_dict['z'], fmt='o-', capsize=6, markersize=6, label='z')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('SMA (pix)')
    ax.set_ylabel('Annulus Mean Intensity (counts/pix)')
    ax.set_title(f'{cln} Isophote Annuli Mean Intensities')

    # Top x-axis using arcsec values
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_arcsec[0], sma_vals_arcsec[-1])
    ax_top.set_xlabel('SMA (arcsec)')

    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_isophote_intensities_ylog_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()






    fig, ax = plt.subplots(figsize=(8, 6))

    # Primary plot (bottom x-axis pix)
    # ax.semilogx(sma_vals_pix, annular_means_g, color=color_dict['g'], marker='o', label='g')
    # ax.semilogx(sma_vals_pix, annular_means_r, color=color_dict['r'], marker='o', label='r')
    # ax.semilogx(sma_vals_pix, annular_means_i, color=color_dict['i'], marker='o', label='i')

    ax.errorbar(sma_vals_pix, annular_means_g, yerr=annuli_mean_error_pix['g'], color=color_dict['g'], fmt='o-', capsize=6, markersize=6, label='g')
    ax.errorbar(sma_vals_pix, annular_means_r, yerr=annuli_mean_error_pix['r'], color=color_dict['r'], fmt='o-', capsize=6, markersize=6, label='r')
    ax.errorbar(sma_vals_pix, annular_means_i, yerr=annuli_mean_error_pix['i'], color=color_dict['i'], fmt='o-', capsize=6, markersize=6, label='i')
    ax.errorbar(sma_vals_pix, annular_means_z, yerr=annuli_mean_error_pix['z'], color=color_dict['z'], fmt='o-', capsize=6, markersize=6, label='z')

    ax.set_xscale('log')

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    ax.set_ylim(-1e-1,1e-1) 

    ax.set_xlabel('SMA (pix)')
    ax.set_ylabel('Annulus Mean Intensity (counts/pix)')
    ax.set_title(f'{cln} Isophote Annuli Mean Intensities')

    # Top x-axis using arcsec values
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_arcsec[0], sma_vals_arcsec[-1])
    ax_top.set_xlabel('SMA (arcsec)')

    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_isophote_intensities_lin_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()




    # Surface brightness 

    # Annular means is already dividing by area, so I just need to 
    # convert annular means to arcsec^-2
    # Converting intensities/counts to surface brightness
    annular_means_arcsecs_g = np.array(annular_means_g)/(arcsec_per_pix_decam)**2
    mu_g = -2.5*np.log10(annular_means_arcsecs_g) + 27

    annular_means_arcsecs_r = np.array(annular_means_r)/(arcsec_per_pix_decam)**2
    mu_r = -2.5*np.log10(annular_means_arcsecs_r) + 27

    annular_means_arcsecs_i = np.array(annular_means_i)/(arcsec_per_pix_decam)**2
    mu_i = -2.5*np.log10(annular_means_arcsecs_i) + 27

    annular_means_arcsecs_z = np.array(annular_means_z)/(arcsec_per_pix_decam)**2
    mu_z = -2.5*np.log10(annular_means_arcsecs_z) + 27

    annular_means_arcsec_dict = {'g': annular_means_arcsecs_g, 'r': annular_means_arcsecs_r, 'i': annular_means_arcsecs_i, 'z': annular_means_arcsecs_z}
    mu_dict = {'g': mu_g, 'r': mu_r, 'i': mu_i , 'z': mu_z}

    # annuli_mean_error_arcsec = annuli_mean_error_pix/(arcsec_per_pix_decam)**2
    annuli_mean_err_arcsec = {band: np.abs(annuli_mean_error_pix[band]/(arcsec_per_pix_decam)**2) for band in annuli_mean_error_pix}

    mu_err_dict = {}
    for band in band_list:
        mu_err_dict[band] = np.abs((2.5/np.log(10))*annuli_mean_err_arcsec[band]/annular_means_arcsec_dict[band])




    # Nearest magnitude to kron radius
    # print(f"SMA values: {sma_vals_arcsec}")

    # rkron_sma_idx_g = np.abs(sma_vals_arcsec - rkron_g_arcsec).argmin()
    # rkron_sma_idx_r = np.abs(sma_vals_arcsec - rkron_r_arcsec).argmin()
    # rkron_sma_idx_i = np.abs(sma_vals_arcsec - rkron_i_arcsec).argmin()

    # rkron_sma_closest_val_g = sma_vals_arcsec[rkron_sma_idx_g]
    # rkron_sma_closest_val_r = sma_vals_arcsec[rkron_sma_idx_r]
    # rkron_sma_closest_val_i = sma_vals_arcsec[rkron_sma_idx_i]

    # print(rkron_sma_closest_val_g, rkron_sma_closest_val_r, rkron_sma_closest_val_i)

    # print(f"SB mag nearest to rkron in g band: {mu_g[rkron_sma_idx_g]}")
    # print(f"SB mag nearest to rkron in r band: {mu_r[rkron_sma_idx_r]}")
    # print(f"SB mag nearest to rkron in i band: {mu_i[rkron_sma_idx_i]}")





    # SMA vals for 21 and 24.5 mag in r-band
    start_idx = int(np.ceil(0.05*len(sma_vals_arcsec))) #ignore the first few values due to noise
    # start_idx = 3 #ignore the first few values due to noise
    sb_lower_idx_r = np.nanargmin(np.abs(mu_dict['r'][start_idx:] - 21)) + start_idx #Ignore the NaNs
    sb_upper_idx_r = np.nanargmin(np.abs(mu_dict['r'][start_idx:] - 24.5)) + start_idx

    sb_lower_closest_val_r = mu_r[sb_lower_idx_r]
    sb_upper_closest_val_r = mu_r[sb_upper_idx_r]

    sb_sma_lower_r = sma_vals_arcsec[sb_lower_idx_r]
    sb_sma_upper_r = sma_vals_arcsec[sb_upper_idx_r]

    print(f"SB nearest to 21st mag in r: {sb_lower_closest_val_r}")
    print(f"SMA nearest to 21st mag in r: {sb_sma_lower_r} arcsec")

    print(f"SB nearest to 24.5th mag in r: {sb_upper_closest_val_r}")
    print(f"SMA nearest to 24.5th mag in r: {sb_sma_upper_r} arcsec")




    # Plotting surface brightness. This plot is redundant so I commented out 

    # fig, ax = plt.subplots(figsize=(8, 6))
    # # ax.semilogx(sma_vals_arcsec,mu_g, color=color_dict['g'], marker='+', label = 'g')
    # # ax.semilogx(sma_vals_arcsec,mu_r, color=color_dict['r'], marker='+', label = 'r')
    # # ax.semilogx(sma_vals_arcsec,mu_i, color=color_dict['i'], marker='+', label = 'i')

    # ax.errorbar(sma_vals_arcsec, mu_g, yerr=mu_err_dict['g'], color=color_dict['g'], fmt='o-', capsize=3, markersize=3, label='g')
    # ax.errorbar(sma_vals_arcsec, mu_r, yerr=mu_err_dict['r'], color=color_dict['r'], fmt='o-', capsize=3, markersize=3, label='r')
    # ax.errorbar(sma_vals_arcsec, mu_i, yerr=mu_err_dict['i'], color=color_dict['i'], fmt='o-', capsize=3, markersize=3, label='i')

    # ax.set_xscale('log')

    # ax.set_ylim(17.5, 30.5) #Fix y limits

    # # For plotting sma vals at 21st/25th mag in r band
    # ax.axvspan(sb_sma_lower_r, sb_sma_upper_r, alpha=0.2, color = 'b')


    # # plt.axvline(x = 1, color = 'gray', label = 'axvline - full height')
    # ax.set_xlabel('SMA (arcsec)')
    # ax.set_ylabel(r'$\mu_0$ (mag/arcsec$^2$)')

    # # Top x-axis with kpc
    # ax_top = ax.twiny()
    # ax_top.set_xscale('log')
    # ax_top.set_xlim(sma_vals_kpc[0], sma_vals_kpc[-1])
    # ax_top.set_xlabel('SMA (kpc)')

    # ax.invert_yaxis()
    # ax.legend()
    # ax.set_title(f'{cln} BCG + ICL Surface Brightness')
    # plt.tight_layout()
    # plt.savefig(f'{output_folder}/{cln}_surface_brightness_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    # plt.close()





    #------------------------------------------------------------------------------------------------------------------------
    # Colors
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Calculating colors...")


    # Colors
    gr_color = mu_dict['g'] - mu_dict['r'] #Bluer - redder
    ri_color = mu_dict['r'] - mu_dict['i'] 
    iz_color = mu_dict['i'] - mu_dict['z']

    gr_color_err = np.sqrt(mu_err_dict['g']**2 + mu_err_dict['r']**2)
    ri_color_err = np.sqrt(mu_err_dict['r']**2 + mu_err_dict['i']**2)
    iz_color_err = np.sqrt(mu_err_dict['i']**2 + mu_err_dict['z']**2)


    # WANT TO BIN THE COLORS, BUT NEED TO WAIT UNTIL I HAVE ERRORS. I HAVE THE ERRORS NOW




    # Plotting colors
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.semilogx(sma_vals_arcsec,gr_color, 'go', label = r'$(g - r)_0$')
    # ax.semilogx(sma_vals_arcsec,ri_color, 'r+', label = r'$(r - i)_0$')

    ax.errorbar(sma_vals_arcsec, gr_color, yerr=gr_color_err, color=color_dict['g'], fmt='o', capsize=6, markersize=6, mec='black',label=r'$(g - r)_0$')
    ax.errorbar(sma_vals_arcsec, ri_color, yerr=ri_color_err, color=color_dict['r'], fmt='o', capsize=6, markersize=6, mec='black',label=r'$(r - i)_0$')
    ax.errorbar(sma_vals_arcsec, iz_color, yerr=iz_color_err, color=color_dict['i'], fmt='o', capsize=6, markersize=6, mec='black',label=r'$(i - z)_0$')

    ax.set_xscale('log')
    ax.set_ylim(-5, 5) 

    # For plotting sma vals at 21st/25th mag in r band
    # ax.axvspan(sb_sma_lower_r, sb_sma_upper_r, alpha=0.2, color = 'b')


    ax.set_xlabel('SMA (arcsec)')
    # ax.set_ylabel(r'$\mu_0$ (mag/arcsec$^2$)')

    # Top x-axis with kpc
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_kpc[0], sma_vals_kpc[-1])
    ax_top.set_xlabel('SMA (kpc)')

    # ax.invert_yaxis()
    ax.legend()
    ax.set_title(f'{cln} Color Profile')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_color_profile_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()








    #------------------------------------------------------------------------------------------------------------------------
    # Sersic Image 
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Fitting Sersic profile...")

    # Indices of SMA to sample in Sersic profile

    #Using A85 bounds
    # sersic_sampling_idx = [index for index,value in enumerate(sma_vals_arcsec) 
    #                        if value > lower_bound_arcsec and value < upper_bound_arcsec]

    # Using r-band cutoff bounds
    sersic_sampling_idx = [index for index,value in enumerate(sma_vals_arcsec) 
                        if value > sb_sma_lower_r and value < sb_sma_upper_r]

    # Sersic parameter ranges to use for fitting
    # fit_params={'n':4.0,'amplitude':38,'r_eff':40,'bounds':{'r_eff':[0,1e5]},'fixed':{'n':True}}
    fit_params={'n':4.0,'amplitude':1,'r_eff':40,'bounds':{'n':[1,6], 'amplitude':[1e-2, 2e1], 'r_eff':[1,2e2]},'fixed':{'n':True}}


    ser_prof = Sersic1D(**fit_params) #Might have use different fit_params eventually?
    fitter = TRFLSQFitter(calc_uncertainties=True)

    fitted_model_dict = {}

    for band in band_list:
        fitted_model_dict[band] = fitter(ser_prof, sma_vals_arcsec[sersic_sampling_idx], 
                            annular_means_arcsec_dict[band][sersic_sampling_idx], weights=1/(annuli_mean_err_arcsec[band][sersic_sampling_idx]),
                            estimate_jacobian=True, maxiter=500, acc=1e-06)
        jacobian = fitter.fit_info.jac #size (n_data, n_param)
        residuals = fitter.fit_info.fun #data - model

        print(jacobian)
        print(residuals)

        # dof = len(residuals) - len(fitted_model_dict[band].param_names)
        dof = len(residuals) - len(jacobian[0])
        # print(dof)

        cost = 0.5*np.sum(residuals**2)
        # print(cost)

        s_sq = 2*cost/dof
        # print(s_sq)

        jtj_inv = np.linalg.inv(jacobian.T @ jacobian)
        cov = jtj_inv*s_sq

        # print(jtj_inv)
        # print(cov)

        param_err = np.sqrt(np.diag(cov))
        print(param_err)
        # print(f'model uncertainty: {np.sqrt(jacobian @ cov @ jacobian.T)}')




    # fitted_model_g = fitter(ser_prof, sma_vals_arcsec[sersic_sampling_idx], 
    #                         annular_means_arcsec_dict['g'][sersic_sampling_idx], weights=1/(annuli_mean_err_arcsec['g'][sersic_sampling_idx]),
    #                         estimate_jacobian=True, maxiter=500, acc=1e-06)
    # fitted_model_r = fitter(ser_prof, sma_vals_arcsec[sersic_sampling_idx], 
    #                         annular_means_arcsec_dict['r'][sersic_sampling_idx], weights=1/(annuli_mean_err_arcsec['r'][sersic_sampling_idx]), 
    #                         estimate_jacobian=True, maxiter=500, acc=1e-06)
    # fitted_model_i = fitter(ser_prof, sma_vals_arcsec[sersic_sampling_idx], 
    #                         annular_means_arcsec_dict['i'][sersic_sampling_idx], weights=1/(annuli_mean_err_arcsec['i'][sersic_sampling_idx]), 
    #                         estimate_jacobian=True, maxiter=500, acc=1e-06)
    # fitted_model_z = fitter(ser_prof, sma_vals_arcsec[sersic_sampling_idx], 
    #                         annular_means_arcsec_dict['z'][sersic_sampling_idx], weights=1/(annuli_mean_err_arcsec['z'][sersic_sampling_idx]), 
    #                         estimate_jacobian=True, maxiter=500, acc=1e-06)
    

    print('g Band Sersic parameters')
    print("Amplitude:", fitted_model_dict['g'].amplitude.value)
    print("Effective Radius:", fitted_model_dict['g'].r_eff.value)
    print("Sersic Index (n):", fitted_model_dict['g'].n.value)
    print('')
    print('r Band Sersic parameters')
    print("Amplitude:", fitted_model_dict['r'].amplitude.value)
    print("Effective Radius:", fitted_model_dict['r'].r_eff.value)
    print("Sersic Index (n):", fitted_model_dict['r'].n.value)
    print('')
    print('i Band Sersic parameters')
    print("Amplitude:", fitted_model_dict['i'].amplitude.value)
    print("Effective Radius:", fitted_model_dict['i'].r_eff.value)
    print("Sersic Index (n):", fitted_model_dict['i'].n.value)
    print('')
    print('z Band Sersic parameters')
    print("Amplitude:", fitted_model_dict['z'].amplitude.value)
    print("Effective Radius:", fitted_model_dict['z'].r_eff.value)
    print("Sersic Index (n):", fitted_model_dict['z'].n.value)

    # Sersic intensities converted to SB
    ser_mag_g = -2.5*np.log10(fitted_model_dict['g'](sma_vals_arcsec)) + 27
    ser_mag_r = -2.5*np.log10(fitted_model_dict['r'](sma_vals_arcsec)) + 27
    ser_mag_i = -2.5*np.log10(fitted_model_dict['i'](sma_vals_arcsec)) + 27
    ser_mag_z = -2.5*np.log10(fitted_model_dict['z'](sma_vals_arcsec)) + 27


    # Plotting surface brightness and 1D Sersic
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.semilogx(sma_vals_arcsec,mu_g, color=color_dict['g'], marker='o', markersize = 6, mec='black',label='DECam g')
    # ax.semilogx(sma_vals_arcsec, ser_mag_g, color=color_dict['g'], linestyle='--', label=r'$S\'ersic$ g')

    ax.errorbar(sma_vals_arcsec, mu_g, yerr=mu_err_dict['g'], color=color_dict['g'], fmt='o', capsize=6, markersize=6, mec='black',label='DECam g')
    ax.errorbar(sma_vals_arcsec, ser_mag_g, color=color_dict['g'], linestyle='--', label=r'$S\'ersic$ g')

    # ax.semilogx(sma_vals_arcsec,mu_r, color=color_dict['r'], marker='o', markersize = 6,mec='black', label='DECam r')
    # ax.semilogx(sma_vals_arcsec, ser_mag_r, color=color_dict['r'], linestyle='--', label=r'$S\'ersic$ r')

    ax.errorbar(sma_vals_arcsec, mu_r, yerr=mu_err_dict['r'], color=color_dict['r'], fmt='o', capsize=6, markersize=6, mec='black',label='DECam r')
    ax.errorbar(sma_vals_arcsec, ser_mag_r, color=color_dict['r'], linestyle='--', label=r'$S\'ersic$ r')

    # ax.semilogx(sma_vals_arcsec,mu_i, color=color_dict['i'], marker='o', markersize = 6, mec='black',label='DECam i')
    # ax.semilogx(sma_vals_arcsec, ser_mag_i, color=color_dict['i'], linestyle='--', label=r'$S\'ersic$ i')

    ax.errorbar(sma_vals_arcsec, mu_i, yerr=mu_err_dict['i'], color=color_dict['i'], fmt='o', capsize=6, markersize=6, mec='black',label='DECam i')
    ax.errorbar(sma_vals_arcsec, ser_mag_i, color=color_dict['i'], linestyle='--', label=r'$S\'ersic$ i')

    ax.errorbar(sma_vals_arcsec, mu_z, yerr=mu_err_dict['z'], color=color_dict['z'], fmt='o', capsize=6, markersize=6, mec='black',label='DECam z')
    ax.errorbar(sma_vals_arcsec, ser_mag_z, color=color_dict['z'], linestyle='--', label=r'$S\'ersic$ z')

    # A85-based sampled boundaries
    # ax.axvspan(0, lower_bound_arcsec, alpha=0.2, color = 'r')
    # ax.axvspan(lower_bound_arcsec, upper_bound_arcsec, alpha=0.2, color = 'b')

    ax.set_xscale('log')

    # r-band cutoff-based boundaries
    # ax.axvspan(0, sb_sma_lower_r, alpha=0.2, color = 'r')
    ax.axvspan(sb_sma_lower_r, sb_sma_upper_r, alpha=0.2, color = 'b')

    # For plotting kron radii
    # ax.axvline(x=rkron_g_arcsec, color='g', linestyle='--', linewidth=1.5, label='Reference line')
    # ax.axvline(x=rkron_r_arcsec, color='r', linestyle='--', linewidth=1.5, label='Reference line')
    # ax.axvline(x=rkron_i_arcsec, color='k', linestyle='--', linewidth=1.5, label='Reference line')

    # ax.axvspan(upper_bound_arcsec, 6e2, alpha=0.2, color = 'y') #Not sure what this is for 

    ax.set_xlabel('SMA (arcsec)')
    ax.set_ylabel(r'$\mu_0$ (mag/arcsec$^2$)')
    ax.set_ylim(19, 31) #Fix y limits

    # Top x-axis with kpc
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_kpc[0], sma_vals_kpc[-1])
    ax_top.set_xlabel('SMA (kpc)')

    ax.invert_yaxis()
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(direction="in")
    ax.legend(prop={'size': 12})
    ax.set_title(f'{cln} Isophote SB and Sersic Profiles', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_surface_brightness_1d_sersic_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()





    # Residuals Sersic - mu
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.semilogx(sma_vals_arcsec,ser_mag_g - mu_g, color=color_dict['g'], label=r'$(S\'ersic - \mu_0)_g$')

    # ax.semilogx(sma_vals_arcsec,ser_mag_r - mu_r, color=color_dict['r'], label=r'$(S\'ersic - \mu_0)_r$')

    # ax.semilogx(sma_vals_arcsec,ser_mag_i - mu_i, color=color_dict['i'], label=r'$(S\'ersic - \mu_0)_i$')


    ax.errorbar(sma_vals_arcsec, ser_mag_g - mu_g, yerr=mu_err_dict['g'], color=color_dict['g'], fmt='o-', capsize=6, markersize=6, mec='black',label=r'$(S\'ersic - \mu_0)_g$')
    ax.errorbar(sma_vals_arcsec, ser_mag_r - mu_r, yerr=mu_err_dict['r'], color=color_dict['r'], fmt='o-', capsize=6, markersize=6, mec='black',label=r'$(S\'ersic - \mu_0)_r$')
    ax.errorbar(sma_vals_arcsec, ser_mag_i - mu_i, yerr=mu_err_dict['i'], color=color_dict['i'], fmt='o-', capsize=6, markersize=6, mec='black',label=r'$(S\'ersic - \mu_0)_i$')
    ax.errorbar(sma_vals_arcsec, ser_mag_z - mu_z, yerr=mu_err_dict['z'], color=color_dict['z'], fmt='o-', capsize=6, markersize=6, mec='black',label=r'$(S\'ersic - \mu_0)_z$')
    ax.set_xscale('log')
    
    # r-band cutoff-based boundaries
    # ax.axvspan(0, sb_sma_lower_r, alpha=0.2, color = 'r')
    # ax.axvspan(sb_sma_lower_r, sb_sma_upper_r, alpha=0.2, color = 'b')


    ax.axhline(y=0, color='k', linestyle='--', linewidth=1) #horizontal line marking over/undersubtraction


    ax.set_xlabel('SMA (arcsec)')
    ax.set_ylabel('Residual')

    # Top x-axis with kpc
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_kpc[0], sma_vals_kpc[-1])
    ax_top.set_xlabel('SMA (kpc)')

    # ax.invert_yaxis()
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(direction="in")
    ax.legend(prop={'size': 12})
    ax.set_title(f'{cln} Surface Brightness Residual', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_sb_residuals_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()




    print("------------------------------------")
    print("Generating Sersic image...")



    # Sersic profile is calculated in units of annulars means/arcsec^2. Need to convert back to original means in per pix^2

    ser_2d_g = sersic_to_image(fitted_model_dict['g'](sma_vals_arcsec)*((arcsec_per_pix_decam)**2),
                                                        sma_vals_pix,
                                                        isophotes_g.eps[:-1],
                                                        isophotes_g.pa[:-1],
                                                        bcg_unmsk_g_data.shape,
                                                        np.array([int(y_bcg_ref), int(x_bcg_ref)]),
                                                        fill_value=0)

    ser_2d_r = sersic_to_image(fitted_model_dict['r'](sma_vals_arcsec)*((arcsec_per_pix_decam)**2),
                                                        sma_vals_pix,
                                                        isophotes_r.eps[:-1],
                                                        isophotes_r.pa[:-1],
                                                        bcg_unmsk_r_data.shape,
                                                        np.array([int(y_bcg_ref), int(x_bcg_ref)]),
                                                        fill_value=0)

    ser_2d_i = sersic_to_image(fitted_model_dict['i'](sma_vals_arcsec)*((arcsec_per_pix_decam)**2),
                                                        sma_vals_pix,
                                                        isophotes_i.eps[:-1],
                                                        isophotes_i.pa[:-1],
                                                        bcg_unmsk_i_data.shape,
                                                        np.array([int(y_bcg_ref), int(x_bcg_ref)]),
                                                        fill_value=0)

    ser_2d_z = sersic_to_image(fitted_model_dict['z'](sma_vals_arcsec)*((arcsec_per_pix_decam)**2),
                                                        sma_vals_pix,
                                                        isophotes_z.eps[:-1],
                                                        isophotes_z.pa[:-1],
                                                        bcg_unmsk_z_data.shape,
                                                        np.array([int(y_bcg_ref), int(x_bcg_ref)]),
                                                        fill_value=0)

    # Plotting 2D Sersic

    ser_2d_dict = {'g': ser_2d_g, 'r': ser_2d_r, 'i': ser_2d_i, 'z': ser_2d_z}

    for band in band_list:
        plt.figure()
        plt.imshow(display(ser_2d_dict[band])*255)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.xlim(x_bcg_ref - 1000, x_bcg_ref + 1000)
        plt.ylim(y_bcg_ref - 1000, y_bcg_ref + 1000)
        plt.title(f'{cln} {band}-Band Sersic Image')
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{cln}_{band}_sersic_2d_iter_{iter}.png', dpi=1000, bbox_inches='tight')
        plt.close()


    # Sersic difference images

    ser_diff_g = coadd_g_data - ser_2d_g
    ser_diff_r = coadd_r_data - ser_2d_r
    ser_diff_i = coadd_i_data - ser_2d_i
    ser_diff_z = coadd_z_data - ser_2d_z

    ser_diff_masked_g = bcg_unmsk_g_data - ser_2d_g
    ser_diff_masked_r = bcg_unmsk_r_data - ser_2d_r
    ser_diff_masked_i = bcg_unmsk_i_data - ser_2d_i
    ser_diff_masked_z = bcg_unmsk_z_data - ser_2d_z


    # Plotting difference images

    ser_diff_dict = {'g': ser_diff_g, 'r': ser_diff_r, 'i': ser_diff_i, 'z': ser_diff_z}
    ser_diff_masked_dict = {'g': ser_diff_masked_g, 'r': ser_diff_masked_r, 'i': ser_diff_masked_i, 'z': ser_diff_masked_z}

    for band in band_list:
        plt.figure()
        plt.imshow(display(ser_diff_dict[band])*255)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.xlim(x_bcg_ref - 1000, x_bcg_ref + 1000)
        plt.ylim(y_bcg_ref - 1000, y_bcg_ref + 1000)
        plt.title(f'{cln} {band}-Band Sersic Difference Image')
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{cln}_{band}_sersic_diff_iter_{iter}.png', dpi=1000, bbox_inches='tight')
        plt.close()

        plt.figure()
        cmap = plt.cm.viridis.copy() 
        # cmap.set_bad(color='black')#Setting masked pixels to black
        cmap.set_under(cmap(0))
        plt.imshow((display(ser_diff_masked_dict[band]).filled(fill_value=-1))*255, cmap=cmap, vmin=0)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.xlim(x_bcg_ref - 1000, x_bcg_ref + 1000)
        plt.ylim(y_bcg_ref - 1000, y_bcg_ref + 1000)
        plt.title(f'{cln} {band}-Band Sersic Difference Masked Image')
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{cln}_{band}_sersic_diff_masked_iter_{iter}.png', dpi=1000, bbox_inches='tight')
        plt.close()


    #------------------------------------------------------------------------------------------------------------------------
    # Updated Segmentation Map
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Generating updated seg map...")


    # Run SExtractor, remove bad masks, and update seg map
    mask_g_data = create_new_segmap(ser_diff_g, mask_g_data,x_bcg_ref, y_bcg_ref, band='g', header=coadd_g_header, run_num=iter)
    mask_r_data = create_new_segmap(ser_diff_r, mask_r_data,x_bcg_ref, y_bcg_ref, band='r', header=coadd_r_header, run_num=iter)
    mask_i_data = create_new_segmap(ser_diff_i, mask_i_data,x_bcg_ref, y_bcg_ref, band='i', header=coadd_i_header, run_num=iter)
    mask_z_data = create_new_segmap(ser_diff_z, mask_z_data,x_bcg_ref, y_bcg_ref, band='z', header=coadd_z_header, run_num=iter)

    print(np.count_nonzero(mask_r_data))
    print(len(np.unique(mask_r_data)))

    print(">"*108)
    print(f"END OF ITERATION {iter}...")



print("------------------------------------")
print('Script successfully run!')
