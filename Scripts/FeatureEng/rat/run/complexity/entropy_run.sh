#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N entropy_SYNGAP              
#$ -wd /exports/eddie/scratch/s2864332/SYNGAP_Rat_Data/FeatureEng/complexity      
#$ -o  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/complexity/logs
#$ -e  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/complexity/logs
#$ -l h_rt=00:59:00 
#$ -l h_rss=32G
#$ -l h_vmem=32G
#$ -pe sharedmem 4

. /etc/profile.d/modules.sh

module load anaconda/2024.02
conda activate /exports/eddie/scratch/s2864332/conda/envs/DiagnoseSYNGAP

which python
python --version
echo "PYTHONPATH=$PYTHONPATH"
hostname
date

# Run the program
export PYTHONPATH=/home/s2864332/MySYNGAP/MySYNGAP:/home/s2864332/MySYNGAP/MySYNGAP/ArtifactDetection:$PYTHONPATH

python3 -m DiagnoseSYNGAP.Scripts.FeatureEng.rat.entropy

echo 'Finish run'
