#========================================================================================================================
# Import libraries

from astropy.io import fits
from astropy.visualization import ManualInterval
from astropy.visualization import AsinhStretch
from astropy.nddata import block_reduce
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
plt.rcParams['figure.dpi'] = 1000

import os
import pandas as pd
import subprocess 
import tempfile

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
        inner isophote.
    iso_out : Isophote object
        outer isophote.

    Returns:
    --------
    annulus_flux : float
        Pixel count/intensity within the annulus.
    full_mask : 2D numpy array
        Array of annuli pixel indices. Same shape as input coadd
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

def coadd_unmask_bcg(coadd, mask_arr, bcg_val, band, header, output_folder, save_full_mask_im=True, run_num=1):
    """
    Unmask BCG in coadd image, and mask everything else. Save the mask with BCG unmasked as a fits.
    
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
        fits header information (wcs) to apply to new image
    output_folder : str
        Output folder path
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

    # Save the bcg-unmasked mask as fits
    bcg_unmask_hdu = fits.PrimaryHDU(arr, header=header)
    bcg_unmask_hdu.writeto(f'{output_folder}/{band}_lsp_sex_mask_bcg_unmask_iter_{run_num}.fits', overwrite=True)

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

        # Save fully masked image as fits. Does not work atm
        # hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=np.where(arr != 0,1, mask_arr)).filled(np.nan), header=header)
        # hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=mask_arr).filled(np.nan), header=header)
        # hdu.writeto(f'{output_folder}/fully_masked_{band}_iter_{run_num}.fits', overwrite=True)

    
    # Apply binary mask to coadd image and return. Value 1 means mask
    return ma.masked_array(coadd, mask=arr)
#------------------------------------------------------------------------------------------------------------------------

def display(arr, vmin=-0.1, vmax=250):

    """
    Clip the intensity and stretch the data for better visualization.
    
    Parameters:
    -----------
    arr : 2D numpy array
        The input image.
    vmin, vmax : float
        Clipping bounds. Sets display range. Default range [-0.1, 250].
        
    Returns:
    --------
    arr_2 : 2D numpy array
        Clipped and stretched image
    """

    interval = ManualInterval(vmin, vmax)
    arr_1 = interval(arr)
    stretch = AsinhStretch(0.0004)
    arr_2 = stretch(arr_1)

    return arr_2

#------------------------------------------------------------------------------------------------------------------------

def draw_lsb_contours(coadd, x_bcg, y_bcg, band, cln, output_folder, scale, unmsk_size = 100, superpix_scale=16, sigma=3, run_num=1):
    """
    Draw magnitude contours on smoothed coadd.
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input array/image. Should be ma.
    x_bcg, y_bcg : int
        BCG location.
    band : string
        String label. (e.g. "r", "i")
    cln : str
        Cluster name.
    output_folder : str
        output folder path
    scale : float
        Conversion factor (kpc/pix)
    unmsk_size : int
        Physical size (kpc) for calculating a radius for cutout. Default 100.
    superpix_scale : int
        Superpixel binning factor. Default 16 pix.
    sigma : int
        Standard deviation value to use for gaussian kernel smoothing. Default 3 sigma.
    run_num : int
        Iteration number.

    """
    arcsec_per_pix_super = superpix_scale*(0.263) #multiply by DECam arcsec per pix

    # Separate masked array
    data = coadd.data
    mask = coadd.mask.astype(bool)

    h, w = data.shape #Size of coadd 

    # block data. Ignores masked pixels
    data_blocked = block_reduce(np.where(mask, 0.0, data), block_size=superpix_scale, func=np.sum)

    valid_pix_blocked = block_reduce((~mask).astype(float), block_size=superpix_scale, func=np.sum)

    # Mean per superpixel
    data_blocked = data_blocked/valid_pix_blocked
    data_blocked[valid_pix_blocked == 0] = np.nan

    # Mask block if too many pix masked
    mask_blocked = block_reduce(mask.astype(float), block_size=superpix_scale,func=np.mean)
    mask_blocked = mask_blocked > 0.8  #set a threshold fraction for blocking

    # smooth data and weights
    data_smoothed = gaussian_filter(np.nan_to_num(data_blocked), sigma=sigma)
    weight_smoothed = gaussian_filter((~mask_blocked).astype(float), sigma=sigma)

    # Renormalize
    data_smoothed /= weight_smoothed
    data_smoothed[weight_smoothed == 0] = np.nan

    #convert from flux to magnitude
    data_smoothed_arcsec = data_smoothed/(arcsec_per_pix_super)**2
    data_smoothed_mag = -2.5*np.log10(data_smoothed_arcsec) + 27
    data_smoothed_mag = np.ma.masked_invalid(data_smoothed_mag)

    # convert physical scale radius to pix
    radius = int(unmsk_size/scale)


    # Plotting the contours over the coadd

    #bcg-unmasked full coadd as background
    plt.figure()
    cmap = plt.cm.viridis.copy() 
    cmap.set_under(cmap(0))
    # plt.imshow(1 - display(data_smoothed), origin='lower', cmap='gray')
    # levels=np.arange(2,9),cmap='Blues'
    levels=np.arange(25,31.5, 0.5) #(n,m) -> [n,...,m - 1]
    contour_color_base ='#ffd166'
    # levels=np.arange(2,9),cmap='#ffd166'
    # levels=np.arange(25,31) #(n,m) -> [n,...,m - 1]
    
    base = mcolors.to_rgb(contour_color_base)
    colors = [tuple(np.clip(c * f, 0, 1) for c in base) for f in np.linspace(1.6, 0.4, len(levels))]
    
    plt.imshow((display(coadd).filled(fill_value=-1))*255, origin='lower', cmap=cmap, vmin=0)
    # cs = plt.contour(data_smoothed_mag, levels=[26,27,28,29,30,31,32], extent=[0, w, 0, h], colors=['black','purple','blue','green','orange','red','indianred'], linewidths=1.2)
    cs = plt.contour(data_smoothed_mag, levels=levels, extent=[0, w, 0, h], colors=colors, linewidths=1.2)
    plt.clabel(cs, fmt='%d mag', fontsize=6)
    # plt.colorbar(label=r'$\mu$ (mag/arcsec$^2$)', orientation="vertical") 
    # plt.imshow(display(smoothed_coadd)*255)
    # plt.gca().invert_yaxis()
    # plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.xlim(x_bcg - radius, x_bcg + radius)
    plt.ylim(y_bcg - radius, y_bcg + radius)
    plt.title(fr'{cln} ${band}$-Band SB Contours Iteration {run_num}')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_full_coadd_iter_{run_num}.png', dpi=1000, bbox_inches='tight')
    plt.close()

    # Downsampling coadd as background. Smoothed background coadd
    plt.figure()
    plt.imshow(1 - display(data_smoothed), origin='lower', cmap='gray')
    # original levels for line below: [26,27,28,29,30,31,32], colors=['black','purple','blue','green','orange','red','indianred']
    cs = plt.contour(data_smoothed_mag, levels=levels, colors=colors, linewidths=1.2)
    plt.clabel(cs, fmt='%d mag', fontsize=6)
    # plt.colorbar(label=r'$\mu$ (mag/arcsec$^2$)', orientation="vertical") 
    # plt.imshow(display(smoothed_coadd)*255)
    # plt.gca().invert_yaxis()
    # plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    # plt.xlim(x_bcg_ref - 1000, x_bcg_ref + 1000)
    # plt.ylim(y_bcg_ref - 1000, y_bcg_ref + 1000)
    plt.title(fr'{cln} ${band}$-Band SB Contours Iteration {run_num}')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_iter_{run_num}.png', dpi=1000, bbox_inches='tight')
    plt.close()

    # Full-res background coadd
    plt.figure()
    plt.imshow(1 - display(data), origin='lower', cmap='gray')
    cs = plt.contour(data_smoothed_mag, levels=levels, colors=colors, linewidths=1.2)
    plt.clabel(cs, fmt='%d mag', fontsize=6)
    plt.colorbar(label=r'$\mu$ (mag/arcsec$^2$)', orientation="vertical") 
    # plt.imshow(display(smoothed_coadd)*255)
    # plt.gca().invert_yaxis()
    # plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    # plt.xlim(x_bcg_ref - 1000, x_bcg_ref + 1000)
    # plt.ylim(y_bcg_ref - 1000, y_bcg_ref + 1000)
    plt.title(fr'{cln} ${band}$-Band SB Contours Iteration {run_num}')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sb_contours_fullres_iter_{run_num}.png', dpi=1000, bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------------------------------------------------

def get_photometric_errors(file_path, band):
    """
    Retrieve errors from photometric correction magnitude difference csv. 
    
    Parameters:
    -----------
    file_path : str
        Location of difference csv files.
    band : str
        Band letter. e.g. 'z'
        
    Returns:
    --------
    List
        List of input error floats for coadds, in units of magnitude.
    """

    output_errors = []

    df = pd.read_csv(file_path)

    diff = df[f'{band}_diff'] #Difference values 
    diff = pd.to_numeric(diff, errors='coerce')  # convert blanks to NaN

    # Generate same bins and midpoints as carried out in photometric_correction
    bins = np.linspace(-0.25, 0.25, 50*2+1)

    # Get histogram bin counts and edges
    n, edges = np.histogram(diff, bins=bins) 

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
        output_errors.append(hwhm)

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
        The input ma (masked array/image).
    x0, y0 : int
        Center values from which to fit isophotes.
    max_SMA : int
        max sma radius 

    Returns:
    --------
    IsophoteList object
        object containing info about each fitted isophote.
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
                                sma=5, eps=0.1, pa=start_pa) 
                    
                    # This is performed on the masked coadds
                    el_fixed = Ellipse(coadd, el_geom) 
                    
                    # Fit the isophotes. This returns an IsophoteList object
                    isophotes = el_fixed.fit_image(sma0=start_sma + 5, minsma=start_sma, maxsma=max_SMA, step=0.12, fix_center=True, \
                                            fix_pa=False, fix_eps=False) 
                    
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

def lsp_region_unmask(coadd, mask_arr, y_bcg, x_bcg, band, scale, output_folder, save_full_mask_im=True, unmsk_size = 150, run_num=1):
    """
    Apply LSP mask to its coadd. unmask a region centered on BCG. Save the unmasked region as fits. 
    
    Parameters:
    -----------
    coadd : 2D numpy array
        The input array.
    mask_arr : 2D numpy array
        The input mask.
    y_bcg, x_bcg: int
        BCG coordinates.
    band : str
        Specify which band (e.g. "r", "i")
    scale : float
        Conversion factor (kpc/pix)
    output_folder : str
        Path to output folder
    save_full_mask_png : boolean
        Set whether to save the fully masked png. Default is False.
    unmsk_size : int
        Physical size (kpc) for which to unmask, centered on BCG. Default 150 kpc.
    run_num : int
        Iteration value. Default 1.
        
    Returns:
    --------
    masked array object
        The partially masked coadd image
    """

    # convert physical scale radius to pix
    radius = int(unmsk_size/scale)

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

    # Save an image of the partially-masked coadd
    if save_full_mask_im:

        plt.figure()
        cmap = plt.cm.viridis.copy() 
        cmap.set_under(cmap(0))
        plt.imshow((display(ma.masked_array(coadd, mask=mask_arr_center_unmsk)).filled(fill_value=-1))*255, cmap=cmap)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Intensity", orientation="vertical") 
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.title(f'Region-Unmasked {band}-band Coadd Iteration {run_num}')
        plt.savefig(f'{output_folder}/{band}_coadd_region_unmasked_iter_{run_num}.png', dpi=1000, bbox_inches='tight')
        plt.close()

        # # Save fully masked image as fits. 
        # # hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=np.where(arr != 0,1, mask_arr)).filled(np.nan), header=header)
        # hdu = fits.PrimaryHDU(ma.masked_array(coadd, mask=mask_arr).filled(np.nan), header=header)
        # hdu.writeto(f'{output_folder}/fully_masked_{band}_iter_{run_num}.fits', overwrite=True)

    # Save coadd cutout to run sextractor on
    coadd_cutout = coadd[y_min:y_max, x_min:x_max]
    hdu = fits.PrimaryHDU(coadd_cutout, header=None)
    hdu.writeto(f'{output_folder}/coadd_region_cutout_{band}_iter_{run_num}.fits', overwrite=True)

    
    # Apply binary mask to coadd image and return. Value 1 means mask.
    return ma.masked_array(coadd, mask=mask_arr_center_unmsk)

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
        Search radius in pixels. Default 25 px.
        
    Returns:
    --------
    (x_max, y_max) : (int, int) tuple
        Coordinates of the brightest pixel found within the radius.
    """

    # Get size of image
    ny, nx = coadd.shape

    # Convert input coordinates to integers
    x_bcg = int(np.round(x_bcg).item())
    y_bcg = int(np.round(y_bcg).item())
    
    # Define bounding box for search
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
                   sextractor_folder,
                   run_num=1,
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
    sextractor_folder : str
        Path to sextractor folder.
    run_num : int
        Which iteration (run) of sextractor.
    segmap_name : str
        Filename to save the segmentation image as (default 'seg.fits').
    detect_thresh_param: str
        Sigma value above which is considered detection.
    tag : str
        identification string when exporting files.
    Returns
    -------
    seg_fits : 
        Segmentation map FITS image.
    cat_df : 
        catalog as a pandas df.
    """
    
    config_file=f'{sextractor_folder}/default.sex' 
    param_file=f'{sextractor_folder}/default.param'
    filter_file=f'{sextractor_folder}/default.conv'
    nnw_file=f'{sextractor_folder}/default.nnw'




    # Filename/path to save the segmentation image and catalog as
    segmap_name = f'{sextractor_folder}/seg_{tag}_{band}_{run_num}.fits'
    # cat_name = f'{sextractor_folder}/catalog_{tag}_{band}_{run_num}.cat'

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
        '-STARNNW_NAME', nnw_file,
        '-CATALOG_NAME', '/dev/null',  # Suppress catalog output if not needed
        # '-CATALOG_NAME', cat_name,
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
    # if not os.path.exists(cat_name):
    #     raise RuntimeError(f"SExtractor did not produce a catalog for {band} band.")


    # # Load the segmentation file
    # seg_fits = fits.open(segmap_name)

    # # Attach WCS/header from the input fits
    # if isinstance(input_fits, str):
    #     input_header = fits.getheader(input_fits)
    # else:
    #     if isinstance(input_fits, fits.PrimaryHDU):
    #         input_header = input_fits.header
    #     elif isinstance(input_fits, fits.HDUList):
    #         input_header = input_fits[0].header
    #     else:
    #         input_header = None

    # # Attach header to segmap
    # if input_header is not None:
    #     seg_fits[0].header.update(input_header)
    #     seg_fits.writeto(segmap_name, overwrite=True)


    # return seg_fits

#------------------------------------------------------------------------------------------------------------------------

def sersic_to_image(flux, sma_values, el_list, pa_list, shape, center):
    """
    Generate a 2D image from 1D list of Sersic profile values.

    Parameters:
    -----------
    flux : 1D array 
        Array of intensity values.
    sma_values : list
        Semi-major axis values (pix).
    el_list : 1D array 
        Array of ellipticities (same length as flux array).
    pa_list : 1D array 
        Array of position angles in radians (same length).
    shape : tuple (ny, nx)
        Output image shape.
    center : tuple (y0, x0)
        Center pixel.

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
    
    # Separate tuples
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

    # Interpolate ellipticity and PA
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
    x_rot =  dx*np.cos(pa_rad) + dy*np.sin(pa_rad)
    y_rot = -dx*np.sin(pa_rad) + dy*np.cos(pa_rad)

    # Elliptical radius at each pixel
    r_ellip = np.sqrt(x_rot**2 + (y_rot * a_over_b)**2)


    # Interpolate brightness
    brightness_interp = interp1d(r_sample, flux, kind='linear',
                                 bounds_error=False, fill_value=(flux[0], flux[-1]))
    
    # Interpolate brightness at every pixel
    img = brightness_interp(r_ellip)
    # img = gaussian_filter(img, sigma=1.0)
    
    return img

#------------------------------------------------------------------------------------------------------------------------

def update_mask(region_path, mask, bcg_y, bcg_x, band, sextractor_folder, run_num=1, thresh_param = '1.5'):
    """
    Runs SExtractor on the saved region cutout FITS file, combines and saves updated full mask, and returns mask.

    Parameters
    ----------
    region_path :
        Path to cutout mask.
    mask : arr
        Mask arr.
    y_bcg, x_bcg : int
        BCG coordinates to center region mask.
    band : str
        Specify which band (e.g. "r", "i")
    sextractor_folder : 
        Path to sextractor folder.
    run_num : 
        Which iteration (run) of sextractor. Default 1
    threshold_param : 
        Sigma threshold param for sextractor run. Default 1.5


    Returns
    -------
    mask : 
        Updated 2D mask array
    bcg_region_mask_val : 
        Updated bcg mask value.
    """

    # Run sextractor on unnmasked region fits.

    run_sextractor(region_path, band=band, sextractor_folder=sextractor_folder, run_num=run_num, detect_thresh_param=thresh_param, tag='region')
    
    # Load in sextractor region mask. Get value of region mask at BCG.

    region_mask_path = f'{sextractor_folder}/seg_region_{band}_{run_num}.fits'
    region_mask_data = fits.open(region_mask_path)[0].data

    # make sure bcg mask val is not one. If so, set to something high
    # if region_mask_data[bcg_y, bcg_x] == 1:
    #     np.where(1, 1000, region_mask_data)
    region_mask_data = np.where(region_mask_data == 1, 1000, region_mask_data)

    # combine lsp mask and sextractor mask
    ny, nx = region_mask_data.shape

    y1 = bcg_y - ny//2
    y2 = y1 + ny
    x1 = bcg_x - nx//2
    x2 = x1 + nx

    # mask[bcg_y - ny//2:bcg_y + ny//2, bcg_x - nx//2:bcg_x + nx//2] = region_mask_data
    mask[y1:y2, x1:x2] = region_mask_data
    # mask[y1:y2, x1:x2] += region_mask_data 

    bcg_region_mask_val = mask[bcg_y, bcg_x] 

    return mask, bcg_region_mask_val