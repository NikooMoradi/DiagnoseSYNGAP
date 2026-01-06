#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N power_SYNGAP              
#$ -wd /exports/eddie/scratch/s2864332/SYNGAP_Rat_Data/FeatureEng/power      
#$ -o  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/power/logs
#$ -e  /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/run/power/logs
#$ -l h_rt=00:59:00 
#$ -l h_rss=32G
#$ -l h_vmem=32G
#$ -pe sharedmem 4
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  virtual memory limit of 32 Gbyte: -l h_vmem
#  resident set size memory limie of 32 Gbyte: -l h_rss

# Initialise the environment modules
. /etc/profile.d/modules.sh

# source /exports/eddie/scratch/s2864332/conda/envs/DiagnoseSYNGAP/etc/conda/activate.d
module load anaconda/2024.02
conda activate /exports/eddie/scratch/s2864332/conda/envs/DiagnoseSYNGAP

which python
python --version
echo "PYTHONPATH=$PYTHONPATH"
hostname
date

# Run the program
# export PYTHONPATH=/home/s2864332/MySYNGAP/MySYNGAP:$PYTHONPATH
export PYTHONPATH=/home/s2864332/MySYNGAP/MySYNGAP:/home/s2864332/MySYNGAP/MySYNGAP/ArtifactDetection:$PYTHONPATH

python3 /home/s2864332/MySYNGAP/MySYNGAP/DiagnoseSYNGAP/Scripts/FeatureEng/rat/power.py

echo 'Finish run'
