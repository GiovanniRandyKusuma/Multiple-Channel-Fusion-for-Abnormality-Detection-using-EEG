import logging
import time
from copy import copy
import sys

import numpy as np

from numpy.random import RandomState
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn

import pickle
import deepeegnet
import shalloweegnet
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment 
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)
import matplotlib.pyplot as plt
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.eegnet import EEGNetv4
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity

from dataset import DiagnosisSet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor

log = logging.getLogger(__name__)
log.setLevel('DEBUG')
def plotroc(exp,title):
    exp.model.eval()
    setname = 'test'
    log.info("Compute predictions for {:s}...".format(
        setname))
    dataset = exp.datasets[setname]
    
    preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                        for b in exp.iterator.get_batches(dataset, shuffle=False)]
    
    preds_per_trial = compute_preds_per_trial(
        preds_per_batch, dataset,
        input_time_length=exp.iterator.input_time_length,
        n_stride=exp.iterator.n_preds_per_input)
    mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                            preds_per_trial]
    mean_preds_per_trial = np.array(mean_preds_per_trial)
    #print(mean_preds_per_trial)

    pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
    #print(pred_labels_per_trial)

    assert pred_labels_per_trial.shape == dataset.y.shape
    #print(pred_labels_per_trial.shape )
    #print(dataset.y.shape)
    accuracy = np.mean(pred_labels_per_trial == dataset.y)
    #print(pred_labels_per_trial == dataset.y)
    print('Accuracy')
    print(accuracy)
    misclass = 1 - accuracy
    column_name = "{:s}_misclass".format(setname)
    out = {column_name: float(misclass)}
    y = dataset.y

    n_true_positive = np.sum((y == 1) & (pred_labels_per_trial == 1))
    n_positive = np.sum(y == 1)
    if n_positive > 0:
        sensitivity = n_true_positive / float(n_positive)
    else:
        sensitivity = np.nan
    column_name = "{:s}_n_true_positive".format(setname)
    out.update({column_name: float(n_true_positive)})

    column_name = "{:s}_n_positive".format(setname)
    out.update({column_name: float(n_positive)})
    column_name = "{:s}_sensitivity".format(setname)
    out.update({column_name: float(sensitivity)})

    n_true_negative = np.sum((y == 0) & (pred_labels_per_trial == 0))
    n_negative = np.sum(y == 0)
    if n_negative > 0:
        specificity = n_true_negative / float(n_negative)
    else:
        specificity = np.nan

    column_name = "{:s}_n_true_negative".format(setname)
    out.update({column_name: float(n_true_negative)})

    column_name = "{:s}_n_negative".format(setname)
    out.update({column_name: float(n_negative)})
    column_name = "{:s} accuracy ".format(setname)
    out.update({column_name: float((n_true_negative+n_true_positive)/float( n_positive + n_negative ))})


    column_name = "{:s}_specificity".format(setname)
    out.update({column_name: float(specificity)})
    if (n_negative > 0) and (n_positive > 0):
        auc = roc_auc_score(y, mean_preds_per_trial[:,1])
    
        fpr, tpr, _ = roc_curve(y,  mean_preds_per_trial[:,1])
        
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic\n"+ title)
        plt.legend(loc="lower right")
        plt.savefig(title+'ROC.png')
        plt.close()
    else:
        auc = np.nan
    column_name = "{:s}_auc".format(setname)
    out.update({column_name: float(auc)})
    if n_negative > 0 and (n_positive > 0):

        n_false_negative= n_positive-n_true_positive
        n_false_positive= n_negative-n_true_negative
        recall=sensitivity
        precision = n_true_positive/(n_true_positive+(n_false_positive))
        column_name = "{:s}_F1".format(setname)
        F1 = 2*(recall*precision)/(recall+precision)
        out.update({column_name: float(F1)})
    log.info(out)
def get_title(filename):
    title=""
    if "noclean" in filename:
        if "deep+eegnet" in filename:
            title = "Deep + EEGNet without Line Noise Removed"
        elif "shallow+eegnet" in filename:
            title = "Shallow + EEGNet without Line Noise Removed"
        elif "eegnet" in filename:
            title = "EEGNet without Line Noise Removed"
        elif "shallow" in filename:
            title = "ShallowNet without Line Noise Removed"
        elif "deep" in filename:
            title = "DeepConvNet without Line Noise Removed"
    else:
        if "deep+eegnet" in filename:
            title = "Deep + EEGNet with Line Noise Removed"
        elif "shallow+eegnet" in filename:
            title = "Shallow + EEGNet with Line Noise Removed"
        elif "eegnet" in filename:
            title = "EEGNet with Line Noise Removed"
        elif "shallow" in filename:
            title = "ShallowNet with Line Noise Removed"
        elif "deep" in filename:
            title = "DeepConvNet with Line Noise Removed"
    return title


def evalmodel(exp):
    exp.model.eval()
    for setname in ('train', 'valid', 'test'):
        log.info("Compute predictions for {:s}...".format(
            setname))
        dataset = exp.datasets[setname]
        if config.cuda:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                               for b in exp.iterator.get_batches(dataset, shuffle=False)]
        else:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0])))
                               for b in exp.iterator.get_batches(dataset, shuffle=False)]
        preds_per_trial = compute_preds_per_trial(
            preds_per_batch, dataset,
            input_time_length=exp.iterator.input_time_length,
            n_stride=exp.iterator.n_preds_per_input)
        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)
        #print(mean_preds_per_trial)

        pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
        #print(pred_labels_per_trial)

        assert pred_labels_per_trial.shape == dataset.y.shape
        #print(pred_labels_per_trial.shape )
        #print(dataset.y.shape)
        accuracy = np.mean(pred_labels_per_trial == dataset.y)
        #print(pred_labels_per_trial == dataset.y)
        print('Accuracy')
        print(accuracy)
        misclass = 1 - accuracy
        column_name = "{:s}_misclass".format(setname)
        out = {column_name: float(misclass)}
        y = dataset.y

        n_true_positive = np.sum((y == 1) & (pred_labels_per_trial == 1))
        n_positive = np.sum(y == 1)
        if n_positive > 0:
            sensitivity = n_true_positive / float(n_positive)
        else:
            sensitivity = np.nan
        column_name = "{:s}_n_true_positive".format(setname)
        out.update({column_name: float(n_true_positive)})

        column_name = "{:s}_n_positive".format(setname)
        out.update({column_name: float(n_positive)})
        column_name = "{:s}_sensitivity".format(setname)
        out.update({column_name: float(sensitivity)})

        n_true_negative = np.sum((y == 0) & (pred_labels_per_trial == 0))
        n_negative = np.sum(y == 0)
        if n_negative > 0:
            specificity = n_true_negative / float(n_negative)
        else:
            specificity = np.nan

        column_name = "{:s}_n_true_negative".format(setname)
        out.update({column_name: float(n_true_negative)})

        column_name = "{:s}_n_negative".format(setname)
        out.update({column_name: float(n_negative)})
        column_name = "{:s} accuracy ".format(setname)
        out.update({column_name: float((n_true_negative+n_true_positive)/float( n_positive + n_negative ))})


        column_name = "{:s}_specificity".format(setname)
        out.update({column_name: float(specificity)})
        if (n_negative > 0) and (n_positive > 0):
            auc = roc_auc_score(y, mean_preds_per_trial[:,1])
        else:
            auc = np.nan
        column_name = "{:s}_auc".format(setname)
        out.update({column_name: float(auc)})
        if n_negative > 0 and (n_positive > 0):

            n_false_negative= n_positive-n_true_positive
            n_false_positive= n_negative-n_true_negative
            recall=sensitivity
            precision = n_true_positive/(n_true_positive+(n_false_positive))
            column_name = "{:s}_F1".format(setname)
            F1 = 2*(recall*precision)/(recall+precision)
            out.update({column_name: float(F1)})
        log.info(out)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def create_set(X, y, inds):
    """
    X list and y nparray
    :return: 
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y,):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.intersect1d(train_inds, test_inds).size == 0
        assert np.intersect1d(valid_inds, test_inds).size == 0
        assert np.array_equal(np.sort(
            np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set


class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.array_equal(np.sort(np.union1d(train_inds, valid_inds)),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set

def run_exp(data_folders,
            n_recordings,
            load_previous_data,
            n_chans,
            max_recording_mins,
            sec_to_cut, duration_recording_mins,
            data_use,
            max_abs_val,
            sampling_freq,
            divisor,
            notes,
            n_folds, i_test_fold,
            shuffle,
            model_name,
            input_time_length, final_conv_length,
            init_lr,
            batch_size, max_epochs,cuda):
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    print(th.cuda.is_available())
    print(th.cuda.device_count())
    print(th.cuda.current_device())
    th.cuda.device(0)
    print(th.cuda.get_device_name(0))
    preproc_functions = []
    #preproc_functions.append(
    #    lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
    #        sec_to_cut * fs)], fs))
            
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))
    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                               (np.clip(data, -max_abs_val, max_abs_val), fs))
    
    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                    filter='kaiser_fast'),sampling_freq))

    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))

    dataset = DiagnosisSet(n_recordings=n_recordings,
                        max_recording_mins=max_recording_mins,
                        preproc_functions=preproc_functions,
                        data_folders=data_folders,sec_to_cut= sec_to_cut,
                        train_or_eval='train',
                        sensor_types=["EEG"])
    

    test_preproc_functions = copy(preproc_functions)

    test_dataset = DiagnosisSet(n_recordings=n_recordings,
                            max_recording_mins=max_recording_mins,
                            preproc_functions=test_preproc_functions,
                            data_folders=data_folders,sec_to_cut=sec_to_cut,
                            train_or_eval='eval',
                            sensor_types=["EEG"])
    
    if(load_previous_data==True):
        with open(data_use+'/X'+'_'+notes+'.pkl', 'rb') as inp:
            X = pickle.load(inp)
        with open(data_use+'/y'+'_'+notes+'.pkl', 'rb') as inp:
            y = pickle.load(inp)
       
    else:
        X,y = dataset.load()
        save_object(X, data_use+'/X'+'_'+notes+'.pkl')
        save_object(y, data_use+'/y'+'_'+notes+'.pkl')
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
  


    max_shape = np.max([list(x.shape) for x in X],
                    axis=0)
    assert max_shape[1] == int(duration_recording_mins *
                             sampling_freq * 60)
   
    if(load_previous_data==True):
        with open(data_use+'/test_X'+'_'+notes+'.pkl', 'rb') as inp:
            test_X = pickle.load(inp)
        with open(data_use+'/test_y'+'_'+notes+'.pkl', 'rb') as inp:
            test_y = pickle.load(inp)
    else:
        test_X, test_y = test_dataset.load()
        save_object(test_X, data_use+'/test_X'+'_'+notes+'.pkl')
        save_object(test_y, data_use+'/test_y'+'_'+notes+'.pkl')
    unique, counts = np.unique(test_y, return_counts=True)
    print(dict(zip(unique, counts)))

    max_shape = np.max([list(x.shape) for x in test_X],
                    axis=0)
    assert max_shape[1] == int(duration_recording_mins *   sampling_freq * 60)


    splitter = TrainValidSplitter(n_folds, i_valid_fold=i_test_fold,
                                    shuffle=shuffle)
    train_set, valid_set = splitter.split(X, y)
    test_set = SignalAndTarget(test_X, test_y)


    del test_X, test_y
    del X,y 

    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    if model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=40,filter_time_length =13,n_filters_spat=40,pool_time_length=35,pool_time_stride=7,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length).create_network()
    elif model_name =="eegnet":
        model = EEGNetv4(in_chans=n_chans, n_classes=n_classes,input_time_length=input_time_length,final_conv_length=final_conv_length).create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         n_filters_time=25,
                         filter_time_length=5,
                         pool_time_length=2,
                         n_filters_spat=25,
                         input_time_length=input_time_length,
                         filter_length_2 = 5,
                         filter_length_3 = 5,
                         filter_length_4 = 5,
                         n_filters_2 = int(25 * 2),
                         n_filters_3 = int(25 * (2 ** 2.0)),
                         n_filters_4 = int(25 * (2 ** 3.0)),
                         final_conv_length=final_conv_length,
                       ).create_network()
    elif model_name =='deep+eegnet':
        model = deepeegnet.DeepEEGNet(in_chans=n_chans, n_classes=n_classes,input_time_length=input_time_length).create_network()
    elif model_name =='shallow+eegnet':
        model = shalloweegnet.ShallowEEGNet(in_chans=n_chans, n_classes=n_classes,input_time_length=input_time_length).create_network()
   
    else:
        assert False, "unknown model name {:s}".format(model_name)
    if model_name !='deep+eegnet_default' and model_name!="shallow+eegnet":
        to_dense_prediction_model(model)
    log.info("Model:\n{:s}".format(str(model)))
    
    if cuda:
        model.cuda()
        
    # determine output size
    test_input = np_to_var(
        np.ones((2, n_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    log.info("In shape: {:s}".format(str(test_input.cpu().data.numpy().shape)))

    out = model(test_input)
    log.info("Out shape: {:s}".format(str(out.cpu().data.numpy().shape)))
    if(config.model_name=='chrononet'):
        n_preds_per_input = out.cpu().data.numpy().shape[1]
    else:
        n_preds_per_input = out.cpu().data.numpy().shape[2]
    log.info("{:d} predictions per input/trial".format(n_preds_per_input))
    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)


    model_constraint = MaxNormDefaultConstraint()
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedDiagnosisMonitor(input_time_length, n_preds_per_input),
                RuntimeMonitor(),]
    stop_criterion = MaxEpochs(max_epochs)
    batch_modifier = None
    run_after_early_stop = False
    if (config.load_model == True):
        model.load_state_dict(th.load('model/'+data_use+model_name+notes+'.pt'))
   
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=run_after_early_stop,
                     batch_modifier=batch_modifier,
                     cuda=cuda)
    if (config.evaluation_only == False):
        exp.run()
        exp.epochs_df.to_pickle("logs"+data_use+model_name+notes+'.pkl')

        
    return exp



if __name__ == "__main__":
    import config
    start_time = time.time()
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    exp = run_exp(
        config.data_folders,
        config.n_recordings,
        config.load_previous_data,
        config.n_chans,
        config.max_recording_mins,
        config.sec_to_cut, config.duration_recording_mins,
        config.data_use,
        config.max_abs_val,
        config.sampling_freq,
        config.divisor,
        config.notes,
        config.n_folds, config.i_test_fold,
        config.shuffle,
        config.model_name,
        config.input_time_length, config.final_conv_length,
        config.init_lr,
        config.batch_size, config.max_epochs,config.cuda,)
    end_time = time.time() 
    run_time = end_time - start_time
    log.info("Experiment runtime: {:.2f} sec".format(run_time))
    

    if(config.evaluation_only== False):
        exp.setup_after_stop_training()
        th.save(exp.model.state_dict(), 'model/'+config.data_use+config.model_name+config.notes+'.pt')
    exp.model.load_state_dict(th.load('model/'+config.data_use+config.model_name+config.notes+'.pt'))
    if(config.plot_roc_auc == True):
        plotroc(exp,get_title(config.model_name+config.notes))
  
    evalmodel(exp)
    
