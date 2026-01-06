#!/bin/sh
#$ -N cross_corr_SYNGAP              
#$ -wd /exports/eddie/scratch/s2864332/SYNGAP_Rat_Data/FeatureEng/connectivity      
#$ -o  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/connectivity/logs
#$ -e  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/connectivity/logs
#$ -l h_rt=00:59:00 
#$ -l h_rss=32G
#$ -l h_vmem=32G
#$ -pe sharedmem 4

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load anaconda/2024.02
conda activate /exports/eddie/scratch/s2864332/conda/envs/DiagnoseSYNGAP

which python
python --version
echo "PYTHONPATH=$PYTHONPATH"
hostname
date


export PYTHONPATH=/home/s2864332/MySYNGAP/MySYNGAP:/home/s2864332/MySYNGAP/MySYNGAP/ArtifactDetection:$PYTHONPATH

python3 -m DiagnoseSYNGAP.Scripts.FeatureEng.rat.connectivity


echo 'Finish run'
