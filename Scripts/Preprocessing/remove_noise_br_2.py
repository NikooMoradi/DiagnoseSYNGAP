import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/s2864332/MySYNGAP/DiagnoseSYNGAP/Scripts/Preprocessing')

from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from constants import (
    SYNGAP_baseline_start,
    SYNGAP_baseline_end,
    channel_variables,
    seizure_free_IDs,
    seizure_two_files,
    SYNGAP_1_ls,
    SYNGAP_2_ls
)

# Paths used in preprocessing
directory_path = '/home/s2864332/SYNGAP_Rat_Data/formatted/numpyformat_baseline/'
seizure_br_path = '/home/melissa/PREPROCESSING/SYNGAP1/csv_seizures/'
clean_br_path = '/home/s2864332/SYNGAP_Rat_Data/Preprocessed/clean_brain_states/'
filtered_data_path = '/home/s2864332/SYNGAP_Rat_Data/Preprocessed/filtered_data/'

# Brainstate number to analyze (usually wake = 0)
br_number = 0


# =====================================================
#  HELPER: SAVING FUNCTIONS
# =====================================================

def save_clean_brain_states(clean_br_path, animal_id, clean_br_1, clean_br_2=None):
    """Save BL1 and (if present) BL2 cleaned brainstate files."""
    clean_br_1.to_pickle(clean_br_path + f"{animal_id}_BL1.pkl")
    print(f"\t\t\tSaved: {animal_id}_BL1.pkl")

    if clean_br_2 is not None:
        clean_br_2.to_pickle(clean_br_path + f"{animal_id}_BL2.pkl")
        print(f"\t\t\tSaved: {animal_id}_BL2.pkl")


def save_filtered_data(save_path, animal_id, filtered_data, suffix="bandpass_filtered"):
    """Saves the bandpass-filtered EEG data to a numpy file."""
    save_file = os.path.join(save_path, f'{animal_id}{suffix}.npy')
    np.save(save_file, filtered_data)
    print(f"\t\t\tSaved filtered data: {save_file}")

# TODO : Complete this function to save clean epoch indices for power analysis
# def save_clean_indices(save_path, animal_id, clean_br, br_number):
#     """Saves a minimal CSV of the clean epoch indices, matching the expected format."""
#     # The 'clean' epochs are those that are NOT marked as artifact (3, 4, 5, 6)
#     clean_indices_df = clean_br[~clean_br['brainstate'].isin([3, 4, 5, 6])].copy()
    
#     # Add Animal_ID for downstream merging, as required by power_analysis_class
#     clean_indices_df['Animal_ID'] = animal_id 
    
#     # Save with the specific filename expected by power_analysis_class.py
#     clean_indices_df[['Epoch', 'Animal_ID']].to_csv(
#         os.path.join(save_path, f'{animal_id}_clean_power.csv'),
#         index=False
#     )
#     print(f"\tSaved clean indices: {animal_id}_clean_power.csv")


# =====================================================
#  Helper: FULL PROCESSING PIPELINE FOR ONE DATASET
# =====================================================

def process_data(noise_filter, bandpass_filtered_data, clean_br, br_number, animal_id):
    """
    Full preprocessing pipeline for one baseline file (BL1 or BL2):
        - detect packet loss
        - compute adaptive noise thresholds
        - detect noisy epochs
        - detect harmonic artifacts
    """

    # 1) Detect packet loss
    print("\t\t→ Detecting packet loss (State 6 = Packet Loss)")
    clean_br, packet_loss_idx = noise_filter.find_packetloss_indices(bandpass_filtered_data, loss_thresh=50)

    # 2) Compute noise thresholds
    print("\t\t→ Calculating noise thresholds")
    slope_thresh, int_thresh = noise_filter.calc_noise_thresh(bandpass_filtered_data, packet_loss_idx)

    # # 3) Power-based noise detection
    # print("\t\t→ Detecting noisy epochs (State 5 = Noise)")
    # _, noise_indices = noise_filter.power_calc_noise(
    #     bandpass_filtered_data,
    #     slope_thresh=slope_thresh,
    #     int_thresh=int_thresh,
    #     clean_br=clean_br,
    #     br_number=br_number
    # )

    # # Label those epochs
    # # print("\t\t→ Labeling noisy epochs in brainstate (State 5 = Noise)")
    # clean_br.loc[noise_indices, "brainstate"] = 5

    # # 4) Harmonic noise detection
    # print("\t\t→ Detecting harmonic artifacts (State 3 = Artifact/Harmonic)")
    # harmonic_indices = HarmonicsFilter(
    #     filtered_data=bandpass_filtered_data,
    #     br_state_file=clean_br,
    #     br_state_num=br_number,
    #     noise_array=noise_indices
    # ).harmonics_algo()

    # # print("\t\t→ Labeling harmonic artifacts in brainstate (State 3 = Artifact/Harmonic)")
    # clean_br.loc[harmonic_indices, "brainstate"] = 3

    # return clean_br


# =====================================================
#  TWO-FILE ANIMALS (BL1 + BL2)
# =====================================================

def preprocess_data_2_animals(animal_ids, save_clean_br=True, save_filtered=True):
    for animal_id in animal_ids:
        print("\n==========================")
        print(f"Processing {animal_id} (BL1 + BL2)")
        print("==========================")

        print("→ Loading files")
        lf = LoadFiles(directory_path, animal_id)
        data_1, data_2, br_1, br_2 = lf.load_two_analysis_files(
            start_times_dict=SYNGAP_baseline_start,
            end_times_dict=SYNGAP_baseline_end
        )

        # Remove seizures
        if animal_id not in seizure_free_IDs:
            print("→ Removing seizure epochs (State 4 = Seizure)")
            seiz_1 = pd.read_csv(seizure_br_path + f"{animal_id}_BL1_Seizures.csv")
            seiz_2 = pd.read_csv(seizure_br_path + f"{animal_id}_BL2_Seizures.csv")
            br_1 = remove_seizure_epochs(br_1, seiz_1)
            br_2 = remove_seizure_epochs(br_2, seiz_2)

        # Filtering
        print("→ Filtering data")
        nf1 = NoiseFilter(data_1, br_1, channelvariables=channel_variables, ch_type="eeg")
        nf2 = NoiseFilter(data_2, br_2, channelvariables=channel_variables, ch_type="eeg")
        filtered_1 = nf1.filter_data_type()
        filtered_2 = nf2.filter_data_type()

        # Full processing pipeline
        print("→ Main preprocessing pipline:")
        clean_br_1 = process_data(nf1, filtered_1, br_1, br_number, animal_id)
        clean_br_2 = process_data(nf2, filtered_2, br_2, br_number, animal_id)

        # # Save
        # if save_clean_br:
        #     print("→ Saving outputs:")
        #     print("\t\t→ Saving cleaned brainstates")
        #     save_clean_brain_states(clean_br_path, animal_id, clean_br_1, clean_br_2)

        # if save_filtered:
        #     print("\t\t→ Saving filtered data")
        #     filtered_full = np.concatenate([filtered_1, filtered_2], axis=1)
        #     save_filtered_data(filtered_data_path, animal_id, filtered_full, suffix="")

        # print("\t\t→ Saving clean epoch indices for power analysis => NOT READY YET!!!!")


# =====================================================
#  ONE-FILE ANIMALS (BL1 only)
# =====================================================

def preprocess_data_1_animal(animal_ids, save_clean_br=True, save_filtered=True):
    for animal_id in animal_ids:
        print("\n==========================")
        print(f"Processing {animal_id} (BL1 only)")
        print("==========================")

        print("→ Loading files")
        lf = LoadFiles(directory_path, animal_id)
        data_1, br_1 = lf.load_one_analysis_file(
            start_times_dict=SYNGAP_baseline_start,
            end_times_dict=SYNGAP_baseline_end
        )

        # Remove seizures
        if animal_id not in seizure_free_IDs:
            print("→ Removing seizure epochs")
            seiz_1 = pd.read_csv(seizure_br_path + f"{animal_id}_BL1_Seizures.csv") # TODO: Ask Lucy about seizure files
            br_1 = remove_seizure_epochs(br_1, seiz_1)

        print("→ Filtering data")
        nf1 = NoiseFilter(data_1, br_1, channelvariables=channel_variables, ch_type="eeg")
        filtered_1 = nf1.filter_data_type()
        
        print("→ Main preprocessing pipline")
        clean_br = process_data(nf1, filtered_1, br_1, br_number, animal_id)

        # # Save
        # if save_clean_br:
        #     print("→ Saving outputs:")
        #     print("\t\t→ Saving cleaned brainstates")
        #     save_clean_brain_states(clean_br_path, animal_id, clean_br)
        # if save_filtered:
        #     print("\t\t→ Saving filtered data")
        #     save_filtered_data(filtered_data_path, animal_id, filtered_1, suffix="")

        # print("\t\t→ Saving clean epoch indices for power analysis => NOY READY YET!!!!")


# =====================================================
#  RUN PREPROCESSING
# =====================================================

# Two-file animals
SYNGAP_2_ls = ['S7063', 'S7064', 'S7069', 'S7070', 'S7071', 'S7072', 'S7083', 'S7086', 'S7091', 'S7098', 'S7101'] # S7096 : problem with size of the recording
preprocess_data_2_animals(SYNGAP_2_ls, save_clean_br=True, save_filtered=False)

# One-file animals
SYNGAP_1_ls = SYNGAP_1_ls = ['S7068', 'S7074', 'S7075', 'S7076', 'S7088', 'S7092'] # S7087 and S7094 : missing seizure file
preprocess_data_1_animal(SYNGAP_1_ls, save_clean_br=True, save_filtered=False)
