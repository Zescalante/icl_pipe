#========================================================================================================================
# Import libraries
import icl_functions

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 1000

import pandas as pd
import sys


#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=6:
    print("Usage: python this_script.py cln" \
    " info_csv photometric_folder sdss_colorterms output_folder") 
    sys.exit(1)

cln = sys.argv[1]
info_csv_path = sys.argv[2]
zp_corr_path = sys.argv[3]
sdss_ct_path = sys.argv[4]
output_folder = sys.argv[5]

#========================================================================================================================
#========================================================================================================================
# FUNCTIONS
#========================================================================================================================
#========================================================================================================================

def polyerr(p,x,xerr):
    # return np.sqrt( (p[1]*xerr)**2 + (2*p[2]*x*xerr)**2 + (3*p[3]*(x**2)*xerr)**2 )
    return abs(p[1] + 2*p[2]*x + 3*p[3]*(x**2))*xerr

#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

color_dict = {'g': 'dodgerblue', 'r': 'tomato', 'i': 'purple', 'z': 'goldenrod'} 

info_df = pd.read_csv(info_csv_path)

bands = ['g','r','i','z']

# Load in intensity profiles
int_g_df = pd.read_csv(f"{output_folder}/{cln}_g_intensity_profile_iter_3.csv")
int_r_df = pd.read_csv(f"{output_folder}/{cln}_r_intensity_profile_iter_3.csv")
int_i_df = pd.read_csv(f"{output_folder}/{cln}_i_intensity_profile_iter_3.csv")
int_z_df = pd.read_csv(f"{output_folder}/{cln}_z_intensity_profile_iter_3.csv")

int_df = {'g': int_g_df, 'r': int_r_df, 'i': int_i_df, 'z': int_z_df}

#========================================================================================================================
# ZERO POINT CORRECTION
#========================================================================================================================

# loading zp-correction
zp_correction_df = pd.read_csv(zp_corr_path)

# compute observed mags/magerrs and apply zp correction 
mag_dict = {band: -2.5*np.log10((int_df[band]['mean_int_back_sub'])/(info_df['arcsec_per_pix'][0]**2)) + 27 for band in bands}
magz_dict = {band: mag_dict[band] - zp_correction_df[band][0] for band in bands}
magerr_dict = {band: np.abs((2.5*int_df[band]['mean_int_back_sub_err'])/(np.log(10)*int_df[band]['mean_int_back_sub'])) for band in bands}

# Same for the 1D Sersic profile
mag_sersic_dict = {band: -2.5*np.log10((int_df[band]['mean_int_back_sub_Sersic1D'])/(info_df['arcsec_per_pix'][0]**2)) + 27 for band in bands}
magz_sersic_dict = {band: mag_sersic_dict[band] - zp_correction_df[band][0] for band in bands}


# propagate zp-err
magerr_dict = {band: np.sqrt(magerr_dict[band]**2 +  zp_correction_df[band][1]**2) for band in bands}

zp_corr_avg_per_diff =  {band: np.mean(abs(magz_dict[band] - mag_dict[band])/mag_dict[band])*100 for band in bands}

for band in bands:
    print(f"zero-point correction avg percent diff {band}: {zp_corr_avg_per_diff[band]:.2f}%")

#========================================================================================================================
# EXTINCTION CORRECTION
#========================================================================================================================

# Recorded values of A_b/E(B-V)_SFD from Schlafly and Finkbeiner 2011. With R_V = 3.1
a_ebv = {'g': 3.237, 'r': 2.176, 'i': 1.595, 'z': 1.217}

# This is the value to subtract from magnitudes!
ext_corr_mag_dict = {band: a_ebv[band]*info_df['ext_SFD_mean'][0] for band in bands}

# Applying extinction correction
magz_dered_dict = {band: magz_dict[band] - ext_corr_mag_dict[band] for band in bands}
magz_dered_sersic_dict = {band: magz_sersic_dict[band] - ext_corr_mag_dict[band] for band in bands}

ext_corr_avg_per_diff = {band: np.mean(abs(magz_dered_dict[band] - mag_dict[band])/mag_dict[band])*100 for band in bands}

for band in bands:
    print(f"extinction correction avg percent diff {band}: {ext_corr_avg_per_diff[band]:.2f}%")

# #========================================================================================================================
# # K-CORRECTION
# #========================================================================================================================

# #Using a gr-based k-correction?

# # HAVE TO MANUALLY CALCUATE THESE COEFFICIENTS FOR EACH CLUSTER
# # from k-correction calculator. SDSS
# k_coeff_dict = {'g': [0.027, -0.471, 1.458, -0.787], #g-r
#                  'r': [0.04, -0.217, 0.528, -0.268], #g-r
#                    'i': [-0.237, 0.177, 0.147, -0.089], #g-i
#                    'z': [-0.031, -0.196, 0.728, -0.5]} #r-z


# kband = lambda band, c: k_coeff_dict[band][0] + k_coeff_dict[band][1]*c + k_coeff_dict[band][2]*(c**2) + k_coeff_dict[band][3]*(c**3)

# # loading our color-terms for SDSS
# sdss_ct_df = pd.read_csv(sdss_ct_path)

# # saving minus-ct, so m_SDSS = m_DE + ct (??????)
# cband = lambda band, c: sdss_ct_df[band][0] + sdss_ct_df[band][1]*c + sdss_ct_df[band][2]*(c**2) + sdss_ct_df[band][3]*(c**3)

# #Using a gr-based k-correction (maybe not?)
# gr = magz_dered_dict['g'] - magz_dered_dict['r']
# gi = magz_dered_dict['g'] - magz_dered_dict['i']
# rz = magz_dered_dict['r'] - magz_dered_dict['z']

# # print(f'DECam g-r color: {gr}')
# gr_err = np.sqrt(magerr_dict['g']**2 + magerr_dict['r']**2)
# gi_err = np.sqrt(magerr_dict['g']**2 + magerr_dict['i']**2)
# rz_err = np.sqrt(magerr_dict['r']**2 + magerr_dict['z']**2)




# # Plotting DECam colors pre k-correction
# fig, ax = plt.subplots(figsize=(8, 6))

# ax.errorbar(int_df['g']['sma_arcsec'], gr, yerr=gr_err, color=color_dict['g'], fmt='o', capsize=6, markersize=6, mec='black',label=f'DECam g-r')
# ax.errorbar(int_df['g']['sma_arcsec'], gi, yerr=gi_err, color=color_dict['r'], fmt='o', capsize=6, markersize=6, mec='black',label=f'DECam g-i')
# ax.errorbar(int_df['g']['sma_arcsec'], rz, yerr=rz_err, color=color_dict['i'], fmt='o', capsize=6, markersize=6, mec='black',label=f'DECam r-z')

# ax.set_xscale('log')

# ax.set_xlabel('SMA (arcsec)')
# ax.set_ylabel('Color')
# # ax.set_ylim(19, 31) #Fix y limits
# ax.legend(prop={'size': 12})
# ax.set_title(f'{cln} DECam colors (pre k-correction)', fontsize = 15)
# plt.tight_layout()
# plt.savefig(f'{output_folder}/{cln}_DECam_colors_prek.png', dpi=1000, bbox_inches='tight')
# plt.close()




# # convert to SDSS-mags for k-correction. MINUS OR PLUS FOR CBAND? I THINK MINUS
# print(cband('g', gr))
# print(cband('r', gr))
# magz_g_dered_sdss = magz_dered_dict['g'] - cband('g', gr)
# magz_r_dered_sdss = magz_dered_dict['r'] - cband('r', gr)
# magz_i_dered_sdss = magz_dered_dict['i'] - cband('i', gi)
# magz_z_dered_sdss = magz_dered_dict['z'] - cband('z', rz)

# magz_dered_sdss_dict = {'g': magz_g_dered_sdss, 'r': magz_r_dered_sdss, 'i': magz_i_dered_sdss, 'z': magz_z_dered_sdss}


# # print(f"color term g: {cband('g', gr)}")
# # print(f"color term r: {cband('r', gr)}")

# print(f'SDSS g mag: {magz_g_dered_sdss}')
# print(f'SDSS r mag: {magz_r_dered_sdss}')

# # propagate errors from DE -> SDSS mags
# magz_g_dered_sdsserr = np.sqrt(magerr_dict['g']**2 + polyerr(sdss_ct_df['g'], gr, gr_err)**2)
# magz_r_dered_sdsserr = np.sqrt(magerr_dict['r']**2 + polyerr(sdss_ct_df['r'], gr, gr_err)**2)
# magz_i_dered_sdsserr = np.sqrt(magerr_dict['i']**2 + polyerr(sdss_ct_df['i'], gi, gi_err)**2)
# magz_z_dered_sdsserr = np.sqrt(magerr_dict['z']**2 + polyerr(sdss_ct_df['z'], rz, rz_err)**2)


# # Applying k-correction
# gr_sdss = magz_g_dered_sdss - magz_r_dered_sdss
# gi_sdss = magz_g_dered_sdss - magz_i_dered_sdss
# rz_sdss = magz_r_dered_sdss - magz_z_dered_sdss

# color_mapping = {'g': gr_sdss, 'r': gr_sdss, 'i': gi_sdss, 'z': rz_sdss}

# gr_sdss_err = np.sqrt(magz_g_dered_sdsserr**2 + magz_r_dered_sdsserr**2)
# gi_sdss_err = np.sqrt(magz_g_dered_sdsserr**2 + magz_i_dered_sdsserr**2)
# rz_sdss_err = np.sqrt(magz_r_dered_sdsserr**2 + magz_z_dered_sdsserr**2)



# # Plotting SDSS colors pre(?) k-correction
# fig, ax = plt.subplots(figsize=(8, 6))

# ax.errorbar(int_df['g']['sma_arcsec'], gr_sdss, yerr=gr_sdss_err, color=color_dict['g'], fmt='o', capsize=6, markersize=6, mec='black',label=f'SDSS g-r')
# ax.errorbar(int_df['g']['sma_arcsec'], gi_sdss, yerr=gi_sdss_err, color=color_dict['r'], fmt='o', capsize=6, markersize=6, mec='black',label=f'SDSS g-i')
# ax.errorbar(int_df['g']['sma_arcsec'], rz_sdss, yerr=rz_sdss_err, color=color_dict['i'], fmt='o', capsize=6, markersize=6, mec='black',label=f'SDSS r-z')

# ax.set_xscale('log')

# ax.set_xlabel('SMA (arcsec)')
# ax.set_ylabel('Color')
# # ax.set_ylim(19, 31) #Fix y limits
# ax.legend(prop={'size': 12})
# ax.set_title(f'{cln} SDSS colors', fontsize = 15)
# plt.tight_layout()
# plt.savefig(f'{output_folder}/{cln}_SDSS_colors.png', dpi=1000, bbox_inches='tight')
# plt.close()





# color_mapping_err = {'g': gr_sdss_err, 'r': gr_sdss_err, 'i': gi_sdss_err, 'z': rz_sdss_err}


# for band in bands:
#     print(color_mapping[band])
#     print(kband(band, color_mapping[band]))

# mag0_dict = {band: magz_dered_dict[band] + kband(band, color_mapping[band]) for band in bands}
# # mag0_dict = {band: magz_dered_sdss_dict[band] + kband(band, color_mapping[band]) for band in bands}

# # Propagate errors from K-corr
# mag0_err_dict = {band: np.sqrt(magerr_dict[band]**2 + polyerr(k_coeff_dict[band], color_mapping[band], color_mapping_err[band])**2) for band in bands}

# k_corr_avg_per_diff = {band: np.mean(abs(mag0_dict[band] - mag_dict[band])/mag_dict[band])*100 for band in bands}

# for band in bands:
    # print(f"k-correction avg percent diff {band}: {k_corr_avg_per_diff[band]:.2f}%")

# # Saving calibrated mags to a csv
# for band in bands:
#     int_df[band]['sb_mag_calib'] = mag0_dict[band]
#     int_df[band]['sb_mag_calib_err'] = mag0_err_dict[band]
#     int_df[band].to_csv(f'{output_folder}/{cln}_{band}_intensity_profile_final.csv', index=False)



#========================================================================================================================
# COSMOLOGICAL SURFACE BRIGHTNESS DIMMING
#========================================================================================================================

# This is the value to subtract from magnitudes! Same for all bands
sb_dimming_corr_mag = 10*np.log10(1 + info_df['redshift'][0])


# Applying sb dimming corrction
# mag0_dict = {band: mag0_dict[band] - sb_dimming_corr_mag for band in bands}
mag0_dict = {band: magz_dered_dict[band] - sb_dimming_corr_mag for band in bands}
mag0_sersic_dict = {band: magz_dered_sersic_dict[band] - sb_dimming_corr_mag for band in bands}

sb_dimming_avg_per_diff = {band: np.mean(abs(mag0_dict[band] - mag_dict[band])/mag_dict[band])*100 for band in bands}

for band in bands:
    print(f"sb-dimming correction avg percent diff {band}: {sb_dimming_avg_per_diff[band]:.2f}%")

# Saving calibrated mags to a csv
for band in bands:
    int_df[band]['sb_mag_calib'] = mag0_dict[band]
    int_df[band]['sb_sersic_mag_calib'] = mag0_sersic_dict[band]
    # int_df[band]['sb_mag_calib_err'] = mag0_err_dict[band]
    int_df[band]['sb_mag_calib_err'] = magerr_dict[band]
    int_df[band].to_csv(f'{output_folder}/{cln}_{band}_intensity_profile_final.csv', index=False)


