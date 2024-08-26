#!/bin/bash                                                                     
#                                                                               
#SBATCH --job-name=prospectorLMCs   
#SBATCH --array=0-34                                        
#SBATCH --ntasks=1                                                              
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/log.%A_%a.out
#SBATCH --error=logs/log.%A_%a.error 
#SBATCH --partition=cpu-q                                       

conda activate prospector

python demoLMC_params.py --objid=${SLURM_ARRAY_TASK_ID} --dynesty --add_duste --outfile=test${SLURM_ARRAY_TASK_ID}
