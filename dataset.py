
from ctypes.wintypes import HPALETTE
import logging
import re
from types import NoneType
import numpy as np
import glob
import os.path
import torch#for testing gpu 
import mne
import config
from pyprep.removeTrend import removeTrend
 
from pyprep import PrepPipeline
log = logging.getLogger(__name__)


def session_key(file_name):
    """ sort the file name by session """
    return re.findall(r'(s\d{2})', file_name)


def natural_key(file_name):
    """ provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None
           for token in re.split(r'(\d+)', file_name)]
    return key

def time_key(file_name):
    """ provides a time-based sorting key """
    splits = file_name.split('/')
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])

    return date_id + session_id + recording_id


def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)
    return file_paths     #RETURN TO STOP SORTING

    if key == 'time':
        return sorted(file_paths, key=time_key)

    elif key == 'natural':
        return sorted(file_paths, key=natural_key)

def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        print(file_path)
  
        if config.format_use=="edf":
            edf_file = mne.io.read_raw_edf(file_path, verbose='error')
        elif config.format_use=="set":
            edf_file = mne.io.read_raw_eeglab(file_path, verbose='error')
        
    except ValueError:
        return None, None, None, None, None, None

    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    # TODO: return rec object?
    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration


def get_recording_length(file_path):
    """ some recordings were that huge that simply opening them with mne caused the program to crash. therefore, open
    the edf as bytes and only read the header. parse the duration from there and check if the file can safely be opened
    :param file_path: path of the directory
    :return: the duration of the recording
    """
    raw = mne.io.read_raw_fif(file_path)

  

    return raw.n_times/raw.info['sfreq']/60




def load_data_edf(fname, preproc_functions,sec_to_cut, sensor_types=['EEG']):
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)
    import config
  
    #log.info("Load data..."+fname)
    if n_sec <=config.duration_recording_mins*60 + 60:
        return None
    ##edit to get on gpu
    torch.cuda.set_device(0)
    montagee = mne.channels.make_standard_montage("standard_1020")
    cnt.load_data()
    new_channel_name= ['A1', 'A2', 'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1',
                        'Fp2', 'Fz', 'O1', 'O2',
                        'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
    selected_ch_names = []
    if 'EEG' in sensor_types:
        wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                        'FP2', 'FZ', 'O1', 'O2',
                        'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

        for wanted_part in wanted_elecs:
            wanted_found_name = []
            for ch_name in cnt.ch_names:
                if wanted_part.lower() in ch_name.lower():
                    wanted_found_name.append(ch_name)
              
   
            if len(wanted_found_name) == 1:
                selected_ch_names.append(wanted_found_name[0])
            else:
                wanted_found_name.sort(key=len)
                selected_ch_names.append(wanted_found_name[0])

    
    #RENAME CHANNEL ACCORDING TO MANTAGE
    mapping = dict(zip(selected_ch_names, new_channel_name))
    cnt.rename_channels(mapping=mapping)
    cnt = cnt.pick_channels(new_channel_name)
    cnt.set_montage(montage=montagee)
    #fig = plt.figure()
    #ax2d = fig.add_subplot(121)
    #ax3d = fig.add_subplot(122, projection='3d')
    #cnt.plot_sensors(ch_type='eeg', axes=ax2d)
    #cnt.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
    #ax3d.view_init(azim=70, elev=15)
    #assert np.array_equal(cnt.ch_names, selected_ch_names)
    n_sensors = 0
    if 'EEG' in sensor_types:
        n_sensors += 21


    assert len(cnt.ch_names)  == n_sensors, (
        "Expected {:d} channel names, got {:d} channel names".format(
            n_sensors, len(cnt.ch_names)))

    cnt.crop(tmin=sec_to_cut,tmax=config.duration_recording_mins*60+sec_to_cut)

    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    #log.info("Preprocessing...")
    import config
    minimum = config.duration_recording_mins *60 *fs
    if data.shape[1] < minimum:
        return None
    
    for fn in preproc_functions:
        #log.info(fn)
        #print(data.shape)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)
    
    return data

def get_all_sorted_file_names_and_labels(train_or_eval, folders):
    all_file_names = []
    for folder in folders:
        full_folder = os.path.join(folder, train_or_eval) + '/'
        log.info("Reading {:s}...".format(full_folder))
        if config.format_use=="edf":
            this_file_names = read_all_file_names(full_folder, '.edf', key='time')
        else:
            this_file_names = read_all_file_names(full_folder, '.set', key='time')
        log.info(".. {:d} files.".format(len(this_file_names)))
        all_file_names.extend(this_file_names)
    log.info("{:d} files in total.".format(len(all_file_names)))
  

    labels = ['/abnormal' in f for f in all_file_names]
    labels = np.array(labels).astype(np.int64)
    return all_file_names, labels


class DiagnosisSet(object):
    def __init__(self, n_recordings, max_recording_mins, preproc_functions,sec_to_cut,
                 data_folders,
                 train_or_eval='train', sensor_types=['EEG'],):
        self.n_recordings = n_recordings
        self.max_recording_mins = max_recording_mins
        self.preproc_functions = preproc_functions
        self.train_or_eval = train_or_eval
        self.sensor_types = sensor_types
        self.data_folders = data_folders
        self.sec_to_cut=sec_to_cut

    def load(self, only_return_labels=False):
        
        log.info("Read file names")
        all_file_names, labels = get_all_sorted_file_names_and_labels(
            train_or_eval=self.train_or_eval,
            folders=self.data_folders,)

        if self.max_recording_mins is not None:
            log.info("Read recording lengths...")
            assert 'train' == self.train_or_eval


            lengths = [get_recording_length(fname) for fname in all_file_names]
            lengths = np.array(lengths)
            mask = lengths < self.max_recording_mins * 60
            cleaned_file_names = np.array(all_file_names)[mask]
            cleaned_labels = labels[mask]
        else:
            cleaned_file_names = np.array(all_file_names)
            cleaned_labels = labels
        if only_return_labels:
            return cleaned_labels
        X = []
        y = []
        n_files = len(cleaned_file_names[:self.n_recordings])
        count=0
        for i_fname, fname in enumerate(cleaned_file_names[:self.n_recordings]):
            log.info("Load {:d} of {:d}".format(i_fname + 1,n_files))

          
            x = load_data_edf(fname=fname, 
                        preproc_functions=self.preproc_functions,
                        sensor_types=self.sensor_types,sec_to_cut= self.sec_to_cut
                       )
           
            if x is None:
                count+=1
                print(count)
                continue
            X.append(x)
            y.append(cleaned_labels[i_fname])
        y = np.array(y)
        return X, y
