#!/bin/bash 
# --- Start of slurm commands -----------

# Request an hour of runtime:
#SBATCH --time=11:59:59

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J icl_measurement_cluster_name

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o slurm_outputs/icl_measurement_cluster_name-%j.out
#SBATCH -e slurm_outputs/icl_measurement_cluster_name-%j.err

#----- End of slurm commands ----

# Run a command

#===========================
# -N 3, -n 5, --mem=50G

# LOAD_LSST="/gpfs/data/idellant/lsst_stack_v26_0_0/loadLSST.bash"
# source ${LOAD_LSST}
# setup lsst_distrib

# Loading SExtractor
module load intel-oneapi-compilers/2023.1.0-5j7s
module load sextractor/2.25.0

# Loading personal research env
module load miniforge3/25.3.0-3
source /oscar/runtime/software/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/etc/profile.d/conda.sh

conda activate research

#===========================
# Variables
parent_folder="/gpfs/data/idellant/zescalan_research/icl_measurement"
script_folder="${parent_folder}/cluster_name/python_scripts/icl_measurement"
output_folder="icl_measurement_output"

sextractor_params_folder="${parent_folder}/A3266_test/SExtractor"
sextractor_folder="SExtractor"

cln="cluster_name"
patches_tag="55-66"
coadd_filename="${patches_tag}_deepCoadd_skycorr.fits"
refcat="reference_catalog"

input_coadds_folder="combine_patch_color_output"
lsp_masks_folder="LSP_masks"
coadd_err_folder="photometric_correction_output"

input_bcg_csv="${parent_folder}/bcg_coords_table.csv"


for band in g r i z; do
    declare "coadd_${band}=${input_coadds_folder}/${cln}_${band}${coadd_filename}"
    # declare "lsp_mask_${band}=${lsp_masks_folder}/${cln}_${band}${coadd_filename}"
    declare "lsp_mask_${band}=${lsp_masks_folder}/${cln}_${band}${patches_tag}_star_updated_lsp_mask.fits"
    declare "mag_diff_err_${band}=${coadd_err_folder}/${cln}_mag_diffs_${refcat}.csv"
    declare "lsp_sex_mask_${band}=${output_folder}/${band}_LSP_SEX_mask_iter_1.fits"
done


#---------------------------
# Script

# Check if output folder exists. If not, create it
[ ! -d ${output_folder} ] && mkdir ${output_folder} 

# Check if SExtractor folder exists. If not, create it

mkdir -p ${sextractor_folder} && cp ${sextractor_params_folder}/default.* ${sextractor_folder}/

echo "STARTING ICL MEASUREMENT SCRIPT..."

echo "---------------------------------------------------------------------------------------------"

echo "QUERYING CLUSTER, FINDING BCG, OTHER METRICS..."

# Run initial script. Pass in cln, coadd_r filepath, bcg_csv file_path, and output folder path
python ${script_folder}/bcg_loc.py ${cln} ${coadd_r} ${input_bcg_csv} ${output_folder}

for band in r g i z; do
    coadd_var="coadd_${band}"

    echo "---------------------------------------------------------------------------------------------"
    
    # Run sextractor to create a star-only mask. Not used for primary anaylysis
    echo "CREATING STAR-ONLY MASK FOR ${band}-BAND..."

    python ${script_folder}/star_mask_ML.py ${cln} ${band} "${!coadd_var}" ${output_folder} ${sextractor_folder}


    echo "---------------------------------------------------------------------------------------------"
    # Expand the mask radius of identified stars
    echo "CREATING EXPANDED-STAR MASK FOR ${band}-BAND..."

    python ${script_folder}/star_mask_expand.py ${cln} ${band} "${!coadd_var}" ${output_folder} ${lsp_masks_folder}


done

for band in r g i z; do
     coadd_var="coadd_${band}"
     lsp_mask_var="lsp_mask_${band}"
     mag_diff_err_var="mag_diff_err_${band}"
     lsp_sex_mask_var="lsp_sex_mask_${band}"

    echo "---------------------------------------------------------------------------------------------"

    echo "INITIAL MASKING AND CALCULATING INPUT ERRORS FOR ${band}-BAND..."

    # Initial masking and coadd input errors
    python ${script_folder}/initial_mask.py ${cln} ${band} "${!coadd_var}" "${!lsp_mask_var}" "${!mag_diff_err_var}" "${output_folder}/${cln}_info.csv" ${sextractor_folder} ${output_folder}

    echo "---------------------------------------------------------------------------------------------"

    echo "RUNNING ITERATION SCRIPT FOR ${band}-BAND..."

    # Iteration script
    python ${script_folder}/iteration.py ${cln} ${band} "${!coadd_var}" "${output_folder}/${band}_lsp_sex_mask_bcg_unmask_iter_1.fits" "${output_folder}/${band}_err_arr.fits" "${output_folder}/${cln}_info.csv" ${sextractor_folder} ${output_folder}
    
    echo "---------------------------------------------------------------------------------------------"

    echo "CALCULATING LIMITING SB FOR ${band}-BAND..."

    # Limiting surface brightness script
    python ${script_folder}/limiting_sb.py ${cln} ${band} "${!coadd_var}" "${output_folder}/${band}_lsp_sex_mask_bcg_unmask_iter_3.fits" "${output_folder}/${band}_err_arr.fits" "${output_folder}/${cln}_info.csv" ${output_folder}

done

echo "---------------------------------------------------------------------------------------------"

echo "APPLYING CORRECTIONS..."

# Corrections script 
python ${script_folder}/corrections.py ${cln} "${output_folder}/${cln}_info.csv" "${coadd_err_folder}/${cln}_matched_residuals.csv" "${parent_folder}/SDSS_factor.csv" ${output_folder}

echo "---------------------------------------------------------------------------------------------"

echo "PLOTTING RESULTS..."

# Corrections script 
python ${script_folder}/results.py ${cln} ${coadd_g} ${coadd_r} ${coadd_i} ${coadd_z} ${output_folder} ${sextractor_folder}

#-----------------------

conda deactivate
