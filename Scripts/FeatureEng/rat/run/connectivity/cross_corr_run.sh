#!/bin/sh
#$ -N cross_corr_SYNGAP_2          
#$ -wd /exports/eddie/scratch/s2864332/SYNGAP_Rat_Data/FeatureEng/connectivity      
#$ -o  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/connectivity/logs/cross_corr
#$ -e  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/connectivity/logs/cross_corr
#$ -l h_rt=05:00:00
#$ -l h_rss=64G
#$ -l h_vmem=64G
#$ -pe sharedmem 4

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load anaconda/2024.02
conda activate /exports/eddie/scratch/s2864332/conda/envs/DiagnoseSYNGAP

export PYTHONPATH=/home/s2864332/MySYNGAP/MySYNGAP:/home/s2864332/MySYNGAP/MySYNGAP/ArtifactDetection:$PYTHONPATH

echo "Running connectivity with:"
echo "  CONNECTIVITY = cross_corr"

python3 -m DiagnoseSYNGAP.Scripts.FeatureEng.rat.connectivity --connectivity cross_corr


echo 'Finish run'
