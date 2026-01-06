import os
import sys
# sys.path.insert(0, "/home/s2864332/MySYNGAP/MySYNGAP")
import artifactdetection as ad
from DiagnoseSYNGAP.Scripts.Preprocessing import (
    SYNGAP_baseline_start,
    SYNGAP_baseline_end,
    analysis_ls,
    SYNGAP_1_ls,
    SYNGAP_2_ls,
)
channel_indices  = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
directory_path = '/exports/eddie/scratch/s2864332/SYNGAP_Rat_Data/formatted_raw/numpyformat_baseline/'

for animal in analysis_ls:
    print(animal)
    for channel in channel_indices:
        print(channel)
        save_folder = f'/exports/eddie/scratch/s2864332/SYNGAP_Rat_Data/FeatureEng/power/channel/channel_{channel}/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        load_files = ad.LoadFiles(directory_path = directory_path, animal_id = animal)
        if animal in SYNGAP_2_ls:
            num_epochs = 34560
            ad.two_files(load_files = load_files, animal = animal, num_epochs = num_epochs, chan_idx = channel,
                  save_folder = save_folder, start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
        if animal in SYNGAP_1_ls:
            num_epochs = 17280
            ad.one_file(load_files = load_files, animal = animal, num_epochs = num_epochs, chan_idx = channel,
                save_folder = save_folder, start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)