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
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.table import Table
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
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=6:
    print("Usage: python this_script.py cln" \
    " band coadd output_folder sextractor_folder") 
    sys.exit(1)

cln = sys.argv[1]
band = sys.argv[2]
coadd_path = sys.argv[3]
output_folder = sys.argv[4]
sextractor_folder = sys.argv[5]

#========================================================================================================================
#========================================================================================================================
# FUNCTIONS
#========================================================================================================================
#========================================================================================================================

def run_sextractor(input_fits, 
                   band,
                   sextractor_folder,
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
    
    config_file=f'{sextractor_folder}/default.sex' 
    param_file=f'{sextractor_folder}/default.param'
    filter_file=f'{sextractor_folder}/default.conv'
    starnnw_file = f'{sextractor_folder}/default.nnw'


    # Filename/path to save the segmentation image and catalog as
    segmap_name = f'{sextractor_folder}/star_seg_{band}.fits'
    cat_name = f'{sextractor_folder}/star_seg_catalog_{band}.cat'

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
        '-STARNNW_NAME', starnnw_file,
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


#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

coadd_data = fits.open(coadd_path)[0].data; coadd_header = fits.open(coadd_path)[0].header

run_sextractor(coadd_path, band = band, sextractor_folder = sextractor_folder)
seg_map_data = fits.open(f'{sextractor_folder}/star_seg_{band}.fits')[0].data; seg_map_header = fits.open(f'{sextractor_folder}/star_seg_{band}.fits')[0].header
cat = Table.read(f'{sextractor_folder}/star_seg_catalog_{band}.cat', format='ascii.sextractor')
cat_df = cat.to_pandas()
# cat_df = pd.read_csv(f'{cln}/SExtractor/star_seg_catalog_{band}.cat', delim_whitespace=True, comment='#')
# print(cat_df['CLASS_STAR'])
cat_gals_df = cat_df[cat_df['CLASS_STAR'] < 0.6] #Filter to only galaxies
gal_ids = cat_gals_df['NUMBER'].values #Get the ids of the galaxies
mask = np.isin(seg_map_data, gal_ids) #Create mask for where the ids are 
seg_map_data[mask] = 0  #Unmask the galaxies

# Save star mask fits
update_mask_hdu = fits.PrimaryHDU(seg_map_data, header=coadd_header)
update_mask_hdu.writeto(f'{sextractor_folder}/star_seg_{band}.fits', overwrite=True)
