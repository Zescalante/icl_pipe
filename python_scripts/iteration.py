#========================================================================================================================
# Import libraries
import icl_functions

from astropy.io import fits
from astropy.modeling.functional_models import Sersic1D
from astropy.modeling.fitting import TRFLSQFitter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_Ellipse
import numpy as np
import numpy.ma as ma
plt.rcParams['figure.dpi'] = 1000

import pandas as pd
import sys


#========================================================================================================================
#========================================================================================================================
# ARGUMENTS
#========================================================================================================================
#========================================================================================================================

if len(sys.argv)!=9:
    print("Usage: python this_script.py cln band coadd mask err_arr info_csv sextractor_folder output_folder") 
    sys.exit(1)

cln = sys.argv[1]
band = sys.argv[2]
coadd_path = sys.argv[3]
mask_path = sys.argv[4]
err_arr_path = sys.argv[5]
info_csv_path = sys.argv[6]
sextractor_folder = sys.argv[7]
output_folder = sys.argv[8]

info_df = pd.read_csv(info_csv_path)
color_dict = {'g': 'dodgerblue', 'r': 'tomato', 'i': 'purple', 'z': 'goldenrod'}  #Band color scheme


#========================================================================================================================
#========================================================================================================================
# SCRIPT
#========================================================================================================================
#========================================================================================================================

# Load in coadd, header, mask, errors

coadd_data = fits.open(coadd_path)[0].data; coadd_header = fits.open(coadd_path)[0].header
mask_data = fits.open(mask_path)[0].data
err_data = fits.open(err_arr_path)[0].data



# We first calculate the largest ellipse to fit within coadd image
ny, nx = coadd_data.shape

# Distance from ellipse center to edges
dist_left = info_df['bcg_x_pix'][0]; dist_right = nx - info_df['bcg_x_pix'][0] 
dist_top = info_df['bcg_y_pix'][0]; dist_bottom = ny - info_df['bcg_y_pix'][0]

# Find max sma
max_sma = 0.98*min(dist_left, dist_right, dist_top, dist_bottom) #extra factor for some cautionary buffer

# Combine data and mask
bcg_unmask_ma = ma.masked_array(coadd_data, mask=mask_data) 

print("------------------------------------")
print("Entering SExtractor loop...")

# Set number of iterations
iterations = 3 

for iter in range(1, iterations + 1):
    print(">"*108)
    print(f"STARTING ITERATION {iter}...")


    #------------------------------------------------------------------------------------------------------------------------
    # Fitting Isophotes 
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Fitting isophotes...")

    #fit isophotes to coadd, with an input center and maximum semi-major axis
    isophotes = icl_functions.isophote_fitter(bcg_unmask_ma, x0=info_df['bcg_x_pix'][0], y0=info_df['bcg_y_pix'][0], max_SMA=max_sma)


    el_pixels = [] #List to store pixel indices in each ellipse aperture


    plt.figure(figsize=(10,8))
    plt.imshow((icl_functions.display(bcg_unmask_ma).filled(fill_value=-1))*255)

    sma_lim = max(iso.sma for iso in isophotes) #Get largest fitted sma. "Limiting sma"


    # plot the isophotes over a masked (except BCG) coadd
    for iso in isophotes:
        e = mpl_Ellipse(xy=(iso.x0, iso.y0), width=2*iso.sma, height=2*iso.sma*(1-iso.eps), \
                    angle=iso.pa*180/np.pi, edgecolor=color_dict[band], facecolor='none')
        plt.gca().add_patch(e) # Plotting the ellipse

    plt.gca().invert_yaxis()
    plt.colorbar(label="Intensity", orientation="vertical") 
    pad = 0.05*sma_lim #padding factor for plot limits. Precautionary
    plt.xlim(info_df['bcg_x_pix'][0] - sma_lim - pad, info_df['bcg_x_pix'][0] + sma_lim + pad)
    plt.ylim(info_df['bcg_y_pix'][0] - sma_lim - pad, info_df['bcg_y_pix'][0] + sma_lim + pad)
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.title(fr'{cln} ${band}$-band Isophotes Iteration {iter}')
    plt.savefig(f'{output_folder}/{cln}_{band}_isophotes_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()


    # Plotting the position angle (PA) and ellipticity (eps) of fitted isophotes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(fr'{cln} ${band}$ Isophote PA and Ellipticity Iteration {iter}', fontsize=14)

    ax1.errorbar([iso.sma for iso in isophotes], [iso.pa for iso in isophotes], yerr=[iso.pa_err for iso in isophotes], color=color_dict[band], fmt='o-', capsize=6, markersize=6, label=band)
    # ax1.set_xscale('log')
    # ax1.set_xlabel('SMA (pix)')
    ax1.set_ylim(0, np.pi)
    ax1.set_ylabel('Isophote PA (rad)', fontsize=12)

    ax2.errorbar([iso.sma for iso in isophotes], [iso.eps for iso in isophotes], yerr=[iso.ellip_err for iso in isophotes], color=color_dict[band], fmt='o-', capsize=6, markersize=6, label=band)
    ax2.set_xscale('log')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('SMA (px)', fontsize=12)
    ax2.set_ylabel('Isophote Ellipticity', fontsize=12)

    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)


    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_isophote_pa_eps_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()

    #------------------------------------------------------------------------------------------------------------------------
    # Surface brightness
    #------------------------------------------------------------------------------------------------------------------------
    print("------------------------------------")
    print("Plotting surface brightness contours...")

    #Generate SB contours over the BCG-unmasked coadd
    icl_functions.draw_lsb_contours(bcg_unmask_ma, x_bcg = info_df['bcg_x_pix'][0], y_bcg = info_df['bcg_y_pix'][0], band=band, cln=cln,
                                    output_folder=output_folder, scale = info_df['kpc_per_pix'][0], unmsk_size = 400, run_num = iter)


    print("------------------------------------")
    print("Calculating surface brightness...")



    # Extract cumulative intensity sums for each isophote
    # So the pixel flux sum in each annulus. This is used for the old method
    annular_sums = np.array([isophotes.tflux_e[i+1] - isophotes.tflux_e[i] \
                    for i in range(len(isophotes.tflux_e)-1)])

    # Extract the rms fluxes and their errors along each isophote
    rms_flux = np.array([isophotes.rms[i] for i in range(len(isophotes.rms))])
    rms_flux_err = np.array([isophotes.int_err[i] for i in range(len(isophotes.int_err))])
    #NEED TO ACCOUNT FOR INTRINSIC ERROR WITH RMS ERROR. USE EllipseSample

    #Lists to store annuli fluxes for all bands
    annuli_fluxes = []
    annuli_idx = []
    annuli_equiv_areas_pix = []

    for i in range(len(isophotes)-1):
        annulus_flux, annulus_pix_idx = icl_functions.annulus_flux_measure(bcg_unmask_ma, isophotes[i], isophotes[i+1])
        annuli_fluxes.append(annulus_flux)
        annuli_idx.append(annulus_pix_idx)
        annuli_equiv_areas_pix.append(np.count_nonzero(annulus_pix_idx))

    # Converting the annuli lists to arrays. annuli_idx is already an array
    annuli_fluxes = np.array(annuli_fluxes) 
    annuli_equiv_areas_pix = np.array(annuli_equiv_areas_pix)



    # Calculating pixel sum errors
    errs = []

    # Loop through the annuli masks indices
    for mask in annuli_idx: 
        pix_err = err_data[mask]
        errs.append(np.sqrt(np.sum(pix_err**2)))

    # store the annuli pix sum errors
    annuli_sum_errs = np.array(errs)


    # Manually calculate the GEOMETRICAL annulus areas pi*a*b. Old method.  
    annular_areas = [((1 - isophotes.eps[i+1])*isophotes.sma[i+1]**2 \
                        - (1 - isophotes.eps[i])*isophotes.sma[i]**2) \
        for i in range(len(isophotes.sma)-1)]
    annular_areas = np.pi*np.array(annular_areas) 


    # Dividing annulus intensity/pix sums by areas to get annular intensity means
    # annular_means = [a/b for a,b in zip(annular_sums,annular_areas)] THIS ONE IS OLD METHOD WITH GEOMETRIC AREAS AND ISO SUBTRACTION FLUX
    annular_means = [a/b for a,b in zip(annuli_fluxes, annuli_equiv_areas_pix)]


    # annular mean errors (pix)
    annuli_mean_error_pix = np.abs(annuli_sum_errs/annuli_equiv_areas_pix)


    # sma_values (x-values)
    sma_vals_pix = np.array([iso.sma for iso in isophotes[:-1]])
    sma_vals_pix_full = np.array([iso.sma for iso in isophotes])

    # Converts SMA from pix to arcsec.
    sma_vals_arcsec = info_df['arcsec_per_pix'][0]*np.array(sma_vals_pix)
    sma_vals_arcsec_full = info_df['arcsec_per_pix'][0]*np.array(sma_vals_pix_full)

    # Get SMA vals in kpc too
    sma_vals_kpc = info_df['kpc_per_arcsec'][0]*np.array(sma_vals_arcsec)


    # Saving intensity data to csv
    int_prof_df = pd.DataFrame({
        'sma_pix': sma_vals_pix,
        'sma_arcsec': sma_vals_arcsec,
        'sma_kpc': sma_vals_kpc,
        'mean_int': annular_means, #In units of counts/pix
        'mean_int_err': annuli_mean_error_pix
    })


    # Plotting annuli mean intensities, pre-background subtraction

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, use_ylim in zip(axes, [False, True]):

        ax.errorbar(sma_vals_pix, annular_means, yerr=annuli_mean_error_pix, color=color_dict[band], fmt='o-', capsize=6, markersize=6, label=band)

        ax.set_xscale('log')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

        ax.set_xlabel('SMA (px)')
        # ax.set_ylabel('Annulus Mean Intensity (counts/px)')

        if use_ylim:
            ax.set_ylim(-1e-1, 1e-1)
        #     ax.set_title('With y-limits')
        # else:
        #     ax.set_title('No y-limits')

        # Top x-axis (arcsec)
        ax_top = ax.twiny()
        ax_top.set_xscale('log')
        ax_top.set_xlim(sma_vals_arcsec[0], sma_vals_arcsec[-1])
        ax_top.set_xlabel('SMA (arcsec)')

    fig.supylabel('Annulus Mean Intensity (counts/px)')

    fig.suptitle(fr'{cln} ${band}$ Isophote Annuli Mean Intensities (No Background Sub) Iteration {iter}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_annuli_mean_intensities_nobacksub_lin_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()


    # Plotting the rms mean fluxes

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, use_ylim in zip(axes, [False, True]):

        ax.errorbar(sma_vals_pix_full, rms_flux, yerr=rms_flux_err, color=color_dict[band], fmt='o-', capsize=6, markersize=6, label=band)

        ax.set_xscale('log')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

        ax.set_xlabel('SMA (px)')
        # ax.set_ylabel('Isophote RMS Intensity (counts/px)')

        if use_ylim:
            ax.set_ylim(-1e-1, 1e-1)
        #     ax.set_title('With y-limits')
        # else:
        #     ax.set_title('No y-limits')

        # Top x-axis (arcsec)
        ax_top = ax.twiny()
        ax_top.set_xscale('log')
        ax_top.set_xlim(sma_vals_arcsec_full[0], sma_vals_arcsec_full[-1])
        ax_top.set_xlabel('SMA (arcsec)')

    fig.supylabel('Isophote RMS Intensity (counts/px)')

    fig.suptitle(fr'{cln} ${band}$ Isophote RMS Intensities (No Background Sub) Iteration {iter}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_rms_intensities_nobacksub_lin_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()




    # BACKGROUND SUBTRACTION

    # CURRENT method of background subtraction: Find minimum value and average n_el points around that value
    n_el = 5
    half_n = n_el//2

    min_int_index = np.argmin(annular_means) #Get mean of n elements around min value


    start_index = max(0, min_int_index - half_n)
    end_index = min(len(annular_means), min_int_index + half_n + 1)

    #Set the background values for the rest of the iterations
    if iter == 1:

        #Regular mean method (all points treated equally)
        # bkg_val = np.mean(annular_means[start_index:end_index])
        # bkg_val_err = np.sqrt(np.sum((annuli_mean_error_pix[start_index:end_index]**2)))/n_el

        # Weighted mean method
        weights = 1.0/annuli_mean_error_pix[start_index:end_index]**2
        bkg_val = np.sum(weights*annular_means[start_index:end_index])/np.sum(weights)
        bkg_val_err = 1.0/np.sqrt(np.sum(weights))

        # Intrinsic scatter error
        scatter_err = np.std(annular_means[start_index:end_index], ddof=1)

    annular_means = annular_means - bkg_val #Subtract the background

    # Propagated error in the annuli mean intensities, after background subtraction
    annuli_mean_error_pix = np.sqrt(annuli_mean_error_pix**2 + bkg_val_err**2 + scatter_err**2)


    #Adding background-subtracted intensities and their errors to df
    int_prof_df['mean_int_back_sub'] = annular_means
    int_prof_df['mean_int_back_sub_err'] = annuli_mean_error_pix




    # Plotting annuli mean intensities, post-background subtraction

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, use_ylim in zip(axes, [False, True]):

        ax.errorbar(sma_vals_pix, annular_means, yerr=annuli_mean_error_pix, color=color_dict[band], fmt='o-', capsize=6, markersize=6, label=band)

        ax.set_xscale('log')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

        ax.set_xlabel('SMA (px)')
        # ax.set_ylabel('Annulus Mean Intensity (counts/px)')

        if use_ylim:
            ax.set_ylim(-1e-1, 1e-1)
        #     ax.set_title('With y-limits')
        # else:
        #     ax.set_title('No y-limits')

        # Top x-axis (arcsec)
        ax_top = ax.twiny()
        ax_top.set_xscale('log')
        ax_top.set_xlim(sma_vals_arcsec[0], sma_vals_arcsec[-1])
        ax_top.set_xlabel('SMA (arcsec)')

    fig.supylabel('Annulus Mean Intensity (counts/px)')

    fig.suptitle(fr'{cln} ${band}$ Isophote Annuli Mean Intensities (With Background Sub) Iteration {iter}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_annuli_mean_intensities_backsub_lin_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()



    # Calculating annulus mean flux S/N and plotting 

    flux_sn = annular_means/annuli_mean_error_pix

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(sma_vals_pix, flux_sn, color=color_dict[band], fmt='o-', capsize=6, markersize=6, mec='black',label=f'DECam {band}')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('SMA (px)')
    ax.set_ylabel(r'SN $f/\sigma_f$')
    # ax.set_ylim(19, 31) #Fix y limits
    ax.legend(prop={'size': 12})

    # Top x-axis (arcsec)
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_kpc[0], sma_vals_kpc[-1])
    ax_top.set_xlabel('SMA (kpc)')

    ax.set_title(fr'{cln} ${band}$-band Annulus Mean Flux SN Iteration {iter}', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_annulus_SN_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()


    # Calculating rms flux S/N and plotting 

    rms_sn = rms_flux/rms_flux_err

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(sma_vals_pix_full, rms_sn, color=color_dict[band], fmt='o-', capsize=6, markersize=6, mec='black',label=f'DECam {band}')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('SMA (px)')
    ax.set_ylabel(r'SN $f/\sigma_f$')
    # ax.set_ylim(19, 31) #Fix y limits
    ax.legend(prop={'size': 12})

    # Top x-axis (arcsec)
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(sma_vals_kpc[0], sma_vals_kpc[-1])
    ax_top.set_xlabel('SMA (kpc)')

    ax.set_title(fr'{cln} ${band}$-band RMS Flux SN Iteration {iter}', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_rms_SN_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()




    # Surface brightness 

    # Annular means is already dividing by area, so I just need to 
    # convert annular means to arcsec^-2
    # Converting intensities/counts to surface brightness
    annular_means_arcsecs = np.array(annular_means)/(info_df['arcsec_per_pix'][0])**2
    mu = -2.5*np.log10(annular_means_arcsecs) + 27

    # annuli_mean_error_arcsec = annuli_mean_error_pix/(arcsec_per_pix_decam)**2
    annuli_mean_err_arcsec = np.abs(annuli_mean_error_pix/(info_df['arcsec_per_pix'][0])**2)

    mu_err = np.abs((2.5/np.log(10))*annuli_mean_err_arcsec/annular_means_arcsecs)


    # CALCULATING FITTING BOUNDS FOR SERSIC PROFILE
    if band != 'r':
        # Load in current iteration's r-band intensity df
        r_int_df = pd.read_csv(f'{output_folder}/{cln}_r_intensity_profile_iter_{iter}.csv')
        r_annular_means_arcsec = np.array(r_int_df['mean_int_back_sub'])/(info_df['arcsec_per_pix'][0])**2
        mu_r_temp = -2.5*np.log10(r_annular_means_arcsec) + 27

        # SMA vals for 21 and 24.5 mag in r-band
        start_idx = int(np.ceil(0.05*len(r_int_df['sma_arcsec']))) #ignore the first few values due to noise

        # start_idx = 3 #ignore the first few values due to noise
        sb_lower_idx_r = np.nanargmin(np.abs(mu_r_temp[start_idx:] - 21)) + start_idx #Ignore the NaNs
        sb_upper_idx_r = np.nanargmin(np.abs(mu_r_temp[start_idx:] - 24.5)) + start_idx

        sb_lower_closest_val_r = mu_r_temp[sb_lower_idx_r]
        sb_upper_closest_val_r = mu_r_temp[sb_upper_idx_r]

        #If I didn't use same isophote-fitting params, I would have to get closest sma val in current band's
        # sma array to those of r-bands sma limit vals. But now they should be the same
        sb_sma_lower_r = r_int_df['sma_arcsec'][sb_lower_idx_r]
        sb_sma_upper_r = r_int_df['sma_arcsec'][sb_upper_idx_r]

    else:
        # SMA vals for 21 and 24.5 mag in r-band
        start_idx = int(np.ceil(0.05*len(sma_vals_arcsec))) #ignore the first few values due to noise
        # start_idx = 3 #ignore the first few values due to noise
        sb_lower_idx_r = np.nanargmin(np.abs(mu[start_idx:] - 21)) + start_idx #Ignore the NaNs
        sb_upper_idx_r = np.nanargmin(np.abs(mu[start_idx:] - 24.5)) + start_idx

        sb_lower_closest_val_r = mu[sb_lower_idx_r]
        sb_upper_closest_val_r = mu[sb_upper_idx_r]

        sb_sma_lower_r = sma_vals_arcsec[sb_lower_idx_r]
        sb_sma_upper_r = sma_vals_arcsec[sb_upper_idx_r]


    #------------------------------------------------------------------------------------------------------------------------
    # Sersic Image 
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Fitting Sersic profile...")

    # Indices of SMA to sample in Sersic profile

    # Using r-band cutoff bounds
    sersic_sampling_idx = [index for index,value in enumerate(sma_vals_arcsec) 
                        if value > sb_sma_lower_r and value < sb_sma_upper_r]

    # Sersic parameter ranges to use for fitting
    fit_params={'n':4.0,'amplitude':1,'r_eff':40,'bounds':{'n':[1,6], 'amplitude':[1e-2, 2e1], 'r_eff':[1,2e2]},'fixed':{'n':True}}

    # initialize the fitter
    ser_prof = Sersic1D(**fit_params)
    fitter = TRFLSQFitter(calc_uncertainties=True)

    # perform the weighted fit
    fitted_model = fitter(ser_prof, sma_vals_arcsec[sersic_sampling_idx], 
                        annular_means_arcsecs[sersic_sampling_idx], weights=1/(annuli_mean_err_arcsec[sersic_sampling_idx]),
                        estimate_jacobian=True, maxiter=500, acc=1e-06)

    # testing calculating parameter errors
    jacobian = fitter.fit_info.jac #size (n_data, n_param)
    residuals = fitter.fit_info.fun #data - model

    # dof = len(residuals) - len(fitted_model_dict[band].param_names)
    dof = len(residuals) - len(jacobian[0])
    cost = 0.5*np.sum(residuals**2)
    s_sq = 2*cost/dof

    jtj_inv = np.linalg.inv(jacobian.T @ jacobian)
    cov = jtj_inv*s_sq

    param_err = np.sqrt(np.diag(cov))




    print(f'{band} Band Sersic parameters')
    print("Amplitude:", fitted_model.amplitude.value)
    print("Effective Radius:", fitted_model.r_eff.value)
    print("Sersic Index (n):", fitted_model.n.value)




    # Adding 1D profile (BCG profile) to intensity df
    int_prof_df['mean_int_back_sub_Sersic1D'] = fitted_model(sma_vals_arcsec)*(info_df['arcsec_per_pix'][0])**2
    # int_prof_df['mean_int_back_sub_int-Sersic_arcsec_err'] = annuli_mean_error_pix

    # export the df
    int_prof_df.to_csv(f'{output_folder}/{cln}_{band}_intensity_profile_iter_{iter}.csv', index=False)




    # Sersic intensities converted to SB
    ser_mag = -2.5*np.log10(fitted_model(sma_vals_arcsec)) + 27


    # Plotting surface brightness and 1D Sersic
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(sma_vals_arcsec, mu, yerr=mu_err, color=color_dict[band], fmt='o', capsize=6, markersize=6, mec='black',label=fr'DECam ${band}$')
    ax.errorbar(sma_vals_arcsec, ser_mag, color=color_dict[band], linestyle='--', label=fr'$S\'{{e}}rsic$ {band}')#label=fr'$S\'{{e}}rsic$ {band}'

    ax.set_xscale('log')

    # r-band cutoff-based boundaries
    # ax.axvspan(0, sb_sma_lower_r, alpha=0.2, color = 'r')
    ax.axvspan(sb_sma_lower_r, sb_sma_upper_r, alpha=0.2, color = 'b')


    ax.set_xlabel('SMA (arcsec)')
    ax.set_ylabel(r'$\mu$ (mag/arcsec$^2$)')
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
    ax.set_title(fr'{cln} ${band}$-band Isophote SB and Sersic Profiles Iteration {iter}', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_surface_brightness_1d_sersic_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()





    # Residuals Sersic - mu
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(sma_vals_arcsec, ser_mag - mu, yerr=mu_err, color=color_dict[band], fmt='o', capsize=6, markersize=6, mec='black',label=fr'$S\'{{e}}rsic - \mu$')
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
    ax.set_title(fr'{cln} ${band}$-band Surface Brightness Residual Iteration {iter}', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sb_residuals_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()


    print("------------------------------------")
    print("Generating Sersic image...")



    # Sersic profile is calculated in units of annulars means/arcsec^2. Need to convert back to original means in per pix^2

    # interpolate 1D flux values into 2D array
    ser_2d = icl_functions.sersic_to_image(fitted_model(sma_vals_arcsec)*((info_df['arcsec_per_pix'][0])**2),
                                                        sma_vals_pix,
                                                        isophotes.eps[:-1],
                                                        isophotes.pa[:-1],
                                                        bcg_unmask_ma.shape,
                                                        np.array([info_df['bcg_y_pix'][0], info_df['bcg_x_pix'][0]])
                                                        )

    # Plotting 2D Sersic

    plt.figure()
    plt.imshow(icl_functions.display(ser_2d)*255)
    plt.gca().invert_yaxis()
    plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.xlim(info_df['bcg_x_pix'][0] - 1000, info_df['bcg_x_pix'][0] + 1000)
    plt.ylim(info_df['bcg_y_pix'][0] - 1000, info_df['bcg_y_pix'][0] + 1000)
    plt.title(fr'{cln} ${band}$-Band Sersic Image Iteration {iter}')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sersic_2d_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()


    # Sersic difference images

    ser_diff = coadd_data - ser_2d
    ser_diff_masked = bcg_unmask_ma - ser_2d


    # Plotting difference images

    plt.figure()
    plt.imshow(icl_functions.display(ser_diff)*255)
    plt.gca().invert_yaxis()
    plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.xlim(info_df['bcg_x_pix'][0] - 1000, info_df['bcg_x_pix'][0] + 1000)
    plt.ylim(info_df['bcg_y_pix'][0] - 1000, info_df['bcg_y_pix'][0] + 1000)
    plt.title(fr'{cln} ${band}$-Band Sersic Difference Image Iteration {iter}')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sersic_diff_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()

    plt.figure()
    cmap = plt.cm.viridis.copy() 
    # cmap.set_bad(color='black')#Setting masked pixels to black
    cmap.set_under(cmap(0))
    plt.imshow((icl_functions.display(ser_diff_masked).filled(fill_value=-1))*255, cmap=cmap, vmin=0)
    plt.gca().invert_yaxis()
    plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.xlim(info_df['bcg_x_pix'][0] - 1000, info_df['bcg_x_pix'][0] + 1000)
    plt.ylim(info_df['bcg_y_pix'][0] - 1000, info_df['bcg_y_pix'][0] + 1000)
    plt.title(fr'{cln} ${band}$-Band Sersic Difference Masked Image Iteration {iter}')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{cln}_{band}_sersic_diff_masked_iter_{iter}.png', dpi=1000, bbox_inches='tight')
    plt.close()



    #------------------------------------------------------------------------------------------------------------------------
    # Updating Segmentation Map
    #------------------------------------------------------------------------------------------------------------------------

    print("------------------------------------")
    print("Generating updated mask...")

    #running sextractor on region and unmasking BCG again
    region_unmsk_data = icl_functions.lsp_region_unmask(ser_diff, mask_data, info_df['bcg_y_pix'][0], info_df['bcg_x_pix'][0], band=band,
                                                        header=coadd_header, scale = info_df['kpc_per_pix'][0], output_folder=output_folder,
                                                        save_full_mask_im=True, unmsk_size=100, run_num=iter + 1)

    updated_mask_data, bcg_mask_val = icl_functions.update_mask(coadd=coadd_data, region_path=f'{output_folder}/coadd_region_cutout_{band}_iter_{iter+1}.fits', mask=mask_data, #do I use bcg unamsked or masked ?
                                            bcg_y=info_df['bcg_y_pix'][0], bcg_x=info_df['bcg_x_pix'][0], band=band, header=coadd_header,
                                            output_folder=output_folder, sextractor_folder=sextractor_folder, run_num=iter + 1, thresh_param = '20.0')


    # Save LSP/SExtractor mask fits
    update_mask_hdu = fits.PrimaryHDU(updated_mask_data, header=coadd_header)
    update_mask_hdu.writeto(f'{output_folder}/{band}_lsp_sex_mask_iter_{iter + 1}.fits', overwrite=True)

    bcg_unmask_ma = icl_functions.coadd_unmask_bcg(coadd_data, updated_mask_data, bcg_mask_val, band, coadd_header, output_folder, save_full_mask_im=True, run_num=iter + 1)

    # display a cutout with only BCG unmasked
    radius = int(300/info_df['kpc_per_pix'][0]) #pixels for cutout of image

    bcg_unmask_ma_stretched = icl_functions.display(bcg_unmask_ma)
    bcg_unmask_ma_stretched = bcg_unmask_ma_stretched.filled(fill_value=-1)
    bcg_unmask_ma_stretched = bcg_unmask_ma_stretched*255

    plt.figure()
    cmap = plt.cm.viridis.copy() 
    # cmap.set_bad(color='black')#Setting masked pixels to black
    cmap.set_under(cmap(0))
    plt.imshow(bcg_unmask_ma_stretched, cmap=cmap, vmin=0)
    # plt.imshow((display(bcg_unmsk_data_dict[band]).filled(fill_value=-1))*255, cmap=cmap, vmin=0)
    plt.gca().invert_yaxis()
    plt.colorbar(label="Intensity", orientation="vertical") 
    plt.xlim(info_df['bcg_x_pix'][0] - radius, info_df['bcg_x_pix'][0] + radius)
    plt.ylim(info_df['bcg_y_pix'][0] - radius, info_df['bcg_y_pix'][0] + radius)
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.title(fr'{cln} Masked ${band}$-band Coadd Iteration {iter + 1}')
    plt.savefig(f'{output_folder}/{cln}_{band}_coadd_bcg_unmasked_iter_{iter + 1}.png', dpi=1000, bbox_inches='tight')
    plt.close()




    print(">"*108)
    print(f"END OF ITERATION {iter}...")


