#FOR VISUALIZING DATA AFTER THE PREPROCESSING

import os.path as op
import numpy as np

import mne

raw = mne.io.read_raw_edf('F:/edf/TUH Dataset/train/abnormal/01_tcp_ar/000/00000016/s004_2012_02_08/00000016_s004_t000.edf',
                          preload=True)

montagee = mne.channels.make_standard_montage("standard_1020")
raw.load_data()
new_channel_name= ['A1', 'A2', 'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1',
                    'Fp2', 'Fz', 'O1', 'O2',
                    'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
selected_ch_names = []

wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                'FP2', 'FZ', 'O1', 'O2',
                'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

for wanted_part in wanted_elecs:
    wanted_found_name = []
    for ch_name in raw.ch_names:
        if wanted_part.lower() in ch_name.lower():#if ' ' + wanted_part + '-' in ch_name:
            wanted_found_name.append(ch_name)
        
    #print(wanted_found_name)####Comment out
    if len(wanted_found_name) == 1:
        selected_ch_names.append(wanted_found_name[0])
    else:
        wanted_found_name.sort(key=len)
        selected_ch_names.append(wanted_found_name[0])


#RENAME CHANNEL ACCORDING TO MANTAGE
mapping = dict(zip(selected_ch_names, new_channel_name))
raw.rename_channels(mapping=mapping)
raw = raw.pick_channels(new_channel_name)
raw.set_montage(montage=montagee)

raw.plot()
#scalings=dict(mag=1e-12, grad=4e-11, eeg=0.00003, eog=150e-6, ecg=5e-4,
#     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
#     resp=1, chpi=1e-4, whitened=1e2)


preproc_functions = []

        
data = (raw.get_data() * 1e6).astype(np.float32)
fs = raw.info['sfreq']
#log.info("Preprocessing...")
preproc_functions.append(lambda data, fs:
                            (np.clip(data, -800, 800), fs))
for fn in preproc_functions:
    #log.info(fn)
    #print(data.shape)
    data, fs = fn(data, fs)
    data = data.astype(np.float32)
    fs = float(fs)
raw._data= data* 1e-6
raw.plot()
print("test")