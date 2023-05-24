# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications
data_folders = [
    'F:/TempData/normal',
    'F:/TempData/abnormal',
    ]
data_use = "TUH" 
n_recordings = None  # set to an integer, if you want to restrict the set size
n_chans = 21 # number of channels being use
max_recording_mins = None # exclude larger recordings from training set
sec_to_cut = 60  # cut away at start of each recording
duration_recording_mins = 10 # how many minutes to use per recording
max_abs_val = 800  # for clipping
sampling_freq = 128 # sampling rate for the data
divisor = 10 # divide signal by this

n_folds = 10 # total folds between the training and validation
i_test_fold = 9
shuffle = True
model_name = 'shallow+eegnet'
#shallow+eegnet for Fusion between ShallowConvNet and EEGNet
#deep+eegnet for Fusion between DeepConvNet and EEGNet
#deep for DeepConvNet
#eegnet for EEGNet
#shallow for ShallowConvNet

input_time_length = 7680 # sampling freq * number of seconds
final_conv_length = 1 # final conv classification layer

init_lr = 1e-3
batch_size = 16 
max_epochs = 1 # until first stop, the continue train on train+valid
cuda = True
load_previous_data = False # if the data is already generated inside the format_use data folder , set False for first time running
evaluation_only= False # evaluation only without training, set False for first time running
notes = "128" #just for notes in the logs files and dataset
format_use = "set" # edf for no Line Removal, set if already do the Line Noise Removal
load_model = False # load the previous model .pt file
plot_roc_auc = True # plot ROC after finish training
