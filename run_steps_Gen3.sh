# ==== #

CLUSTER_NAME='A85'

# NODES=10 # might need to be tweaked for exploratory
# CORES=180 # multiple of 10's only!
# RAM=1300 # multiple of 10 only!
# WALL_TIME=48:00:00 # default time is 2-days, may need to be adjusted for exploratory users

# == VARIABLES == #

AUTO_PIPELINE_DIR="/gpfs/data/idellant/zescalan_research/icl_measurement/automatic_pipeline_gen3_mock" 
TEMPLATE_DIR="${AUTO_PIPELINE_DIR}/processing_step_templates_mock"
CLUSTER_DIR="/gpfs/data/idellant/zescalan_research/icl_measurement/${CLUSTER_NAME}" 
PROCESSING_STEP_DIR="${CLUSTER_DIR}/processing_step"

# == INITIALIZE LSP == #

# source ${LOAD_PIPELINE_PATH}
# setup lsst_distrib

# == FUNCTIONS ==  #

# helper function for printing text
prompt () {

	echo "-------------------------"
	echo $1
	echo "-------------------------"

}

# helper function to tell user a job has been submitted
prompt_wait () {

	prompt "Please wait for the sbatch job to finish! Use 'myq' and 'myjobinfo' to monitor progress."

}

# STEP 0: creates output directories for slurm-scripts and directory for processing_steps

create_output () {

echo "Running STEP 0: create_output"

# initialize .../CLUSTER_NAME
mkdir -p ${CLUSTER_DIR}/processing_step
mkdir -p ${CLUSTER_DIR}/slurm_outputs

cp -a ${AUTO_PIPELINE_DIR}/python_scripts/. ${CLUSTER_DIR}/python_scripts


echo "Making necessary symlinks"

ln -s ~/data/Clusters/gen3_processing/${CLUSTER_NAME}/combine_patch_color_output/ combine_patch_color_output

ln -s ~/data/Clusters/gen3_processing/${CLUSTER_NAME}/masks/ LSP_masks

ln -s ~/data/Clusters/gen3_processing/${CLUSTER_NAME}/photometric_correction_output/ photometric_correction_output

echo "Done!"

}

icl_measurement () {

	REFCAT_INSTRUMENT=$1
    REFCAT_DATA_RELEASE=$2

	echo "Running STEP ??: icl_measurement"


	# pass the cluster name into the template
	sed "s/cluster_name/${CLUSTER_NAME}/g; s/reference_catalog/${REFCAT_INSTRUMENT}_${REFCAT_DATA_RELEASE}/g" ${TEMPLATE_DIR}/icl_measurement_revamp_template.sh > ${PROCESSING_STEP_DIR}/icl_measurement.sh

	# pass the current lsst_pipeline and cluster_dir into the script
	# sed -i "s|load_pipeline_path|${LOAD_PIPELINE_PATH}|g;s|cluster_dir|${CLUSTER_DIR}|g" ${PROCESSING_STEP_DIR}/noao_download_manager.sh

	echo "Submitting to slurm..."
	sbatch ${PROCESSING_STEP_DIR}/icl_measurement.sh
	prompt_wait

	sleep 5s
	
}


create_output


# icl_measurement "sm" "dr4"
