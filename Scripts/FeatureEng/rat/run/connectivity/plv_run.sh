#!/bin/sh
#$ -N PLV_SYNGAP_2_files   
#$ -wd /exports/eddie/scratch/s2864332/SYNGAP_Rat_Data/FeatureEng/connectivity      
#$ -o  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/connectivity/logs/plv
#$ -e  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/connectivity/logs/plv
#$ -l h_rt=10:00:00 
#$ -l h_rss=64G
#$ -l h_vmem=64G
#$ -pe sharedmem 4

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load anaconda/2024.02
conda activate /exports/eddie/scratch/s2864332/conda/envs/DiagnoseSYNGAP


# Run the program
export PYTHONPATH=/home/s2864332/MySYNGAP/MySYNGAP:/home/s2864332/MySYNGAP/MySYNGAP/ArtifactDetection:$PYTHONPATH

python3 -m DiagnoseSYNGAP.Scripts.FeatureEng.rat.connectivity --connectivity phase_lock


echo 'Finish run'

