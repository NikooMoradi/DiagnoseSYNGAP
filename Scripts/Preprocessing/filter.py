import os 
import numpy as np 
import pandas as pd 
import scipy
from scipy import signal 
# from numba import njit

#from parameters import channelvariables


class NoiseFilter:
    """Class to input unfiltered data, returns bandpass filtered data and calculates whether epoch
    is within specified noise threshold.
    """
    
    order = 3
    sampling_rate = 250.4 
    nyquist = sampling_rate/2
    low = 1/nyquist
    high = 100/nyquist

    
    def __init__(self, unfiltered_data, brain_state_file, channelvariables, ch_type):
        self.unfiltered_data = unfiltered_data 
        self.brain_state_file = brain_state_file                        #dataframe with brainstates  
        self.num_epochs = len(brain_state_file)
        self.channel_variables = channelvariables                       #dictionary with channel types and channel variables 
        self.channel_types= channelvariables['channel_types']
        self.channel_numbers = channelvariables['channel_numbers']
        self.ch_type = ch_type                                          #specify channel type to perform calculations on     

        
    def find_packetloss_indices(self, filtered_data, loss_thresh=31000): #NOTE: lost_data_symbol= 32767
        """
        Detects packet-loss epochs by checking if the maximum amplitude 
        in an epoch is below a noise threshold.
        
        Args:
            noise_limit (int or float): amplitude threshold to classify packet loss.
        
        Returns:
            clean_br (DataFrame): brainstate DataFrame (copied).
            packet_loss_idx (list): indices of packet-loss epochs.
        """

        def is_packet_loss(epoch):
            return epoch.max() > loss_thresh
        
        split_data = np.array_split(filtered_data, self.num_epochs, axis=1)

        packet_loss_idx = []
        for idx, epoch in enumerate(split_data):
            if is_packet_loss(epoch):
                packet_loss_idx.append(idx)

        clean_br = self.brain_state_file.copy()
        
        # mark packet-loss epochs as brainstate 6
        clean_br.loc[packet_loss_idx, "brainstate"] = 6

        return clean_br, packet_loss_idx

    def calc_noise_thresh(self, filtered_data, packet_loss_idx):
        """
        Calculates slope and intercept thresholds for noise detection
        using linear regression on Welch power spectra.

        Args:
            packet_loss_idx (list): epoch indices to exclude from threshold estimation.

        Returns:
            slope_thresh (float): lower slope limit (more negative = noise)
            int_thresh (float): upper intercept limit (higher intercept = noise)
        """

        slopes = []
        intercepts = []
        split_epochs = np.array_split(filtered_data, self.num_epochs, axis=1)

        for idx, epoch in enumerate(split_epochs):
            # if idx in packet_loss_idx:
            chan = epoch[0]

            # Welch power spectrum
            freq, power = scipy.signal.welch(
                chan,
                window='hann',
                fs=self.sampling_rate,
                nperseg=1252
            )

            # Linear regression fit
            s, i = np.polyfit(freq, power, 1)

            slopes.append(s)
            intercepts.append(i)
            # else:
            #     pass

        # Convert to arrays
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        # Compute thresholds:
        slope_thresh = slopes.mean()
        int_thresh = intercepts.mean()

        print(f"\t\t\t→ Slope threshold: {slope_thresh}")
        print(f"\t\t\t→ Intercept threshold: {int_thresh}")

        return slope_thresh, int_thresh



    def filter_data_type(self):
         
        def butter_bandpass(data):
            butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype = 'band', analog = False)
            filtered_data = signal.filtfilt(butter_b, butter_a, data)
            return filtered_data
        
        indices = []
        if self.ch_type == 'eeg':
            for idx, ch in enumerate(self.channel_types):
                if ch == 'eeg':
                    indices.append(idx)
        if self.ch_type == 'emg':
            for idx, ch in enumerate(self.channel_types):
                if ch == 'emg':
                        indices.append(idx)
        if self.ch_type == 'all':
            indices = self.channel_numbers
        
        
        
        #Select all, emg, or eeg channel indices to apply bandpass filter                                    
        selected_channels = self.unfiltered_data[indices, :]     
        bandpass_filtered_data=butter_bandpass(data=selected_channels) 
                        
        return bandpass_filtered_data
    
    
    def specify_filter(self, low, high):
        def butter_bandpass(data):
            butter_b, butter_a = signal.butter(self.order, [low/self.nyquist,high/self.nyquist], btype = 'band', analog = False)
            filtered_data = signal.filtfilt(butter_b, butter_a, data)
            return filtered_data
        
        indices = []
        if self.ch_type == 'eeg':
            for idx, ch in enumerate(self.channel_types):
                if ch == 'eeg':
                    indices.append(idx)
        if self.ch_type == 'emg':
            for idx, ch in enumerate(self.channel_types):
                if ch == 'emg':
                        indices.append(idx)
        if self.ch_type == 'all':
            indices = self.channel_numbers
        
        
        
        #Select all, emg, or eeg channel indices to apply bandpass filter                                    
        selected_channels = self.unfiltered_data[indices, :]     
        bandpass_filtered_data=butter_bandpass(data=selected_channels) 
                        
        return bandpass_filtered_data

    
    def power_calc_noise(self, bandpass_filtered_data, slope_thresh, int_thresh, clean_br, br_number):
        
        # def lin_reg_calc(epoch):
        #     noise_array = []
        #     power_array = []

        #     freq, power = scipy.signal.welch(epoch, window = 'hann', fs = 250.4, nperseg = 1252)
        #     slope, intercept = np.polyfit(freq, power, 1)
        #     power_array.append(power)
        #     if intercept > int_thresh or slope < slope_thresh:
        #         noise_array.append(5)
        #     else:
        #         noise_array.append(0)
            
        #     return noise_array, power_array    
        
        
        # def apply_lin_reg(bandpass_filtered_data, clean_br, br_number): 
        #     '''function applies lin_reg_calc function to entire time series and returns two arrays,
        #     one with noise labels and one with power calculation results'''
        #     split_epochs = np.split(bandpass_filtered_data, self.num_epochs, axis = 1)
        #     noisy_indices = []
        #     power_calc = []

        #     # packet_loss = (clean_br.query('brainstate == 6')).index.tolist()
        #     packet_loss = clean_br[clean_br['brainstate'] == 6].index.tolist()
        #     br_calc = clean_br[clean_br['brainstate'] == br_number].index.tolist()
            
        #     for idx, epoch in enumerate(split_epochs):
        #         if idx in packet_loss:
        #             pass
        #         elif idx in br_calc:
        #             channel_arrays = []
        #             for chan in epoch:
        #                     power= lin_reg_calc(chan)
        #                     channel_arrays.append(power)
        #             one_epoch_arrays = np.vstack(channel_arrays)
        #             if one_epoch_arrays[0][0][2] == 5:
        #                 noisy_indices.append(idx)
        #             else:
        #                 one_epoch_power = np.vstack(one_epoch_arrays[1][0])
        #                 power_calc.append(one_epoch_power)
        #         else:
        #             pass
        #     power_array = np.array(power_calc)
        #     noise_array = np.array(noisy_indices)
        #     return power_array, noise_array
        
        def apply_lin_reg(bandpass_filtered_data, clean_br, br_number):
            """Applies linear regression to all epochs and returns:
            - all power arrays (for non-noisy epochs)
            - indices classified as noisy
            """
            n_epochs = len(clean_br)
            # print(
            #     "bandpass_filtered_data shape:", bandpass_filtered_data.shape,
            #     "\nn_epochs:", n_epochs,
            #     "\nremainder:", bandpass_filtered_data.shape[1] % n_epochs
            # )
            split_epochs = np.array_split(bandpass_filtered_data, self.num_epochs, axis=1)

            noisy_indices = []
            power_calc = []

            packet_loss = clean_br[clean_br['brainstate'] == 6].index.tolist()
            br_calc = clean_br[clean_br['brainstate'] == br_number].index.tolist()
            if slope_thresh < -8:
                slope_thresh = -8
            for idx, epoch in enumerate(split_epochs):
                if idx in packet_loss:
                    continue
                elif idx not in br_calc:
                    continue

                # results for all channels for this epoch
                channel_noise = []
                channel_power = []

                for chan in epoch:  # each channel => TODO: probably don't need to do this for each channel, just one or on the mean of channels (chan = epoch[0] or chan = epoch.mean(axis=0))
                    freq, power = scipy.signal.welch(
                        chan, window='hann', fs=250.4, nperseg=1252
                    )
                    slope, intercept = np.polyfit(freq, power, 1)
                    # slope, intercept, power = NoiseFilter.compute_slope_and_intercept(chan, fs=250.4)

                    channel_noise.append(5 if (intercept > int_thresh or slope < slope_thresh) else 0)
                    channel_power.append(power)  # store power array only

                channel_noise = np.array(channel_noise)         # shape (n_channels,)
                channel_power = np.stack(channel_power, axis=0) # shape (n_channels, n_freqs)

                # If ANY channel is noisy → epoch is noisy
                if np.any(channel_noise == 5):
                    noisy_indices.append(idx)
                else:
                    power_calc.append(channel_power)

            return np.array(power_calc, dtype=object), np.array(noisy_indices)

        
        power_array, noise_array = apply_lin_reg(bandpass_filtered_data, clean_br, br_number)
        return power_array, noise_array
    
    
class HarmonicsFilter:
    
    def __init__(self, filtered_data, br_state_file, br_state_num, noise_array):
        self.filtered_data = filtered_data
        self.br_state_file = br_state_file
        self.br_state_num = br_state_num
        self.noise_array = noise_array      #Array with values cleaned from lin reg algo 
        self.num_epochs = len(br_state_file)
    
    def harmonics_algo(self):
        
        '''Function to remove ictal epoch artifacts using z-score moving mean algorithm
        '''
        
        def thresholding_algo( y, lag, threshold, influence):
            signals = np.zeros(len(y))
            filteredY = np.array(y)
            avgFilter = [0]*len(y)
            stdFilter = [0]*len(y)
            avgFilter[lag - 1] = np.mean(y[0:lag])
            stdFilter[lag - 1] = np.std(y[0:lag])
            for i in range(lag, len(y)):
                if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
                    if y[i] > avgFilter[i-1]:
                        signals[i] = 1
                    else:
                        signals[i] = -1

                    filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                    avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                    stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
                else:
                    signals[i] = 0
                    filteredY[i] = y[i]
                    avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                    stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

            signal_calc = np.asarray(signals)
            harmonic_1 = np.mean(signal_calc[25:50])
            harmonic_2 = np.mean(signal_calc[60:85])
            return harmonic_1, harmonic_2
        
    
        split_epochs = np.split(self.filtered_data, self.num_epochs, axis = 1)
        packet_loss = (self.br_state_file.query('brainstate == 6')).index.tolist()
        br_calc = self.br_state_file[self.br_state_file['brainstate'] == self.br_state_num].index.tolist()
        harmonic_indices = []
        for idx, epoch in enumerate(split_epochs):
            if idx in packet_loss:
                pass
            elif idx in br_calc:
                power_calculations = signal.welch(epoch[2], window = 'hann', fs = 250.4, nperseg = 1252)
                harmonic_1, harmonic_2 = thresholding_algo(y = power_calculations[1], lag = 30, threshold = 5, influence = 0)
                if harmonic_1 or harmonic_2 > 0:
                    harmonic_indices.append(idx)
            else:
                pass
        
        total_noise = list(self.noise_array) + list(harmonic_indices)
        
        return total_noise
    
def remove_seizure_epochs(br_normal, seizure_br):
    '''Function returns seizure indices '''
    
    def round_to_multiple(number, multiple):
        return multiple*round(number/multiple)
    
    wake_br = br_normal.loc[br_normal['brainstate'] == 0]
    
    seiz_start = seizure_br['sec_start'].to_numpy()
    seiz_end = seizure_br['sec_end'].to_numpy()
    start_times_5 = [round_to_multiple(i, 5) for i in seiz_start] 
    end_times_5 = [round_to_multiple(i, 5) for i in seiz_end] 
    
    seizure_times = list(start_times_5) + list(end_times_5)
    
    all_ictal_epochs = []
    for seizure_epoch in seizure_times:
        epoch_bins = 5
        preceding_epochs = [seizure_epoch - epoch_bins*5, seizure_epoch - epoch_bins*4, seizure_epoch - epoch_bins*3, seizure_epoch - epoch_bins*2, seizure_epoch - epoch_bins]
        following_epochs = [seizure_epoch + epoch_bins, seizure_epoch + epoch_bins*2, seizure_epoch + epoch_bins*3, seizure_epoch + epoch_bins*4, seizure_epoch + epoch_bins*5 ]
        all_ictal_epochs.extend(preceding_epochs + [seizure_epoch] + following_epochs)
    
    seizure_indices = []
    for epoch_time, epoch_idx in zip(wake_br['start_epoch'].to_numpy(), wake_br.index.to_list()):
        if epoch_time in all_ictal_epochs:
            seizure_indices.append(epoch_idx)
        else:
            pass
        
    br_normal.loc[seizure_indices, 'brainstate'] = 4
    
    return br_normal