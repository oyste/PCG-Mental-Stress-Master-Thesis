import numpy as np
import pandas as pd
import os
import scipy.signal as sg

def load_data(d_type='pcg', 
              pcg_path='pcg', 
              pcg_scatter_path=None, 
              eeg_raw_path='eeg_raw', 
              eeg_filtered_path='eeg_filtered',
              markers_path='markers',
              wanted_recs='all'):

    """
    This function loads data of different types (pcg, markers, eeg_raw, eeg_filtered) from specified
    paths and returns a dictionary of the loaded data.
    
    :param d_type: The type of data to load, which can be 'pcg', 'markers', 'eeg_raw', or
    'eeg_filtered', defaults to pcg (optional)
    :param pcg_path: The path to the directory containing the pcg data files, defaults to pcg (optional)
    :param pcg_scatter_path: The parameter pcg_scatter_path is not used in the function and can be
    removed
    :param eeg_raw_path: The path where the raw EEG data is stored, defaults to eeg_raw (optional)
    :param eeg_filtered_path: The path where the filtered EEG data is stored, defaults to eeg_filtered
    (optional)
    :param markers_path: The path where the markers data is stored, defaults to markers (optional)
    :param wanted_recs: A list of record names that the user wants to load. If set to 'all', all records
    will be loaded, defaults to all (optional)
    :return: a dictionary containing the data of the specified type (pcg, markers, eeg_raw, or
    eeg_filtered) for the specified recordings (either all recordings or a subset specified by the
    wanted_recs parameter).
    """

    if d_type == 'pcg':
        d = {}
        for fname in os.listdir('pcg'):
            if fname.split('.')[0] not in wanted_recs and wanted_recs != 'all': continue
            d[fname.split('.')[0]] = np.load(f'pcg/{fname}')
        return d
    
    elif d_type == 'markers':
        d = {}
        for fname in os.listdir('markers'):
            if fname.split('.')[0] not in wanted_recs and wanted_recs != 'all': continue
            d[fname.split('.')[0]] = np.load(f'markers/{fname}')
        return d
    
    elif d_type == 'eeg_raw':
        d = {}
        for fname in os.listdir('eeg_raw'):
            if fname.split('.')[0] not in wanted_recs and wanted_recs != 'all': continue
            d[fname.split('.')[0]] = np.load(f'eeg_raw/{fname}')
        return d
    
    elif d_type == 'eeg_filtered':
        d = {}
        for fname in os.listdir('eeg_filtered'):
            if fname.split('.')[0] not in wanted_recs and wanted_recs != 'all': continue
            d[fname.split('.')[0]] = np.load(f'eeg_filtered/{fname}')
        return d
    
def load_labels(label_type=None, labels_path='labels', wanted_recs='all'):
    """
    This function loads labels from a pickle file and returns them based on the label type and wanted
    records.
    
    :param label_type: A string indicating the type of label to load (e.g. 'stai', 'pss', 'age',
    'gender')
    :param labels_path: The path to the directory where the labels file is stored, defaults to labels
    (optional)
    :param wanted_recs: This parameter is used to filter the records in the labels dictionary. If set to
    'all', all records are returned. If set to a list of record keys, only the records with those keys
    are returned, defaults to all (optional)
    :return: either a dictionary of labels for a specific label type (if label_type parameter is
    provided), or the entire labels dataframe (if label_type parameter is not provided). If the
    wanted_recs parameter is provided with a list of record IDs, the function filters the labels
    dictionary to only include those records.
    """
    labels = pd.read_pickle('labels.pkl')
    if label_type == 'stai':
        l = labels['stai'].to_dict()
    elif label_type == 'pss':
        l = labels['pss'].to_dict()
    elif label_type == 'age':
        l = labels['age'].to_dict()
    elif label_type == 'gender':
        l =  labels['gender'].to_dict()
    if wanted_recs != 'all':
        for k in l.keys():
            if k not in wanted_recs:
                del l[k]
        return l
    return labels

def vec_norm(v):
    """
    The function calculates the normalized vector of a given vector.
    
    :param v: a vector (list or array) of numerical values
    :return: The function `vec_norm` takes a vector `v` as input and returns the normalized vector. The
    normalization is done by dividing each element of the vector by the maximum absolute value of the
    vector. Therefore, the function returns a normalized vector.
    """
    return v/max(abs(v))

def resample_and_normalize(pcg_dict, from_freq=22050, to_freq=1000):
    """
    This function resamples and normalizes a dictionary of PCG signals from a given frequency to a
    target frequency.
    
    :param pcg_dict: A dictionary containing PCG recordings as values and their corresponding IDs as
    keys
    :param from_freq: The original sampling frequency of the PCG signal, measured in Hz (Hertz),
    defaults to 22050 (optional)
    :param to_freq: The target frequency to which the PCG signals will be resampled, defaults to 1000
    (optional)
    :return: a dictionary `ds` where the keys are the same as the input dictionary `pcg_dict` and the
    values are the resampled and normalized versions of the corresponding values in `pcg_dict`.
    """
    ds = {}
    for k, v in pcg_dict.items():
        ds[k] = vec_norm(sg.resample(v, int(np.floor(len(v)*to_freq/from_freq))))
    return ds

def get_pcg_data_and_labels(to_freq=1000, label_type='stai-3520'):
    """
    This function loads PCG data and corresponding labels, resamples and normalizes the data, and
    returns the data and labels based on the specified label type.
    
    :param to_freq: The desired sampling frequency for the PCG data after resampling and normalization,
    defaults to 1000 (optional)
    :param label_type: The type of label to use for the data. It can be 'stai', 'stai-3520', or
    'stai-3030', defaults to stai-3520 (optional)
    :return: two values: a dictionary containing resampled and normalized PCG data, and a dictionary
    containing labels for the PCG data based on the specified label type.
    """
    pcg_data = load_data(d_type='pcg')
    labels = pd.read_pickle('labels.pkl')
    
    if label_type == 'stai':
        l = labels['STAI']
        l = l.to_dict()
    elif label_type == 'stai-3520':
        l = labels[[a or b for a,b in zip(labels['STAI'] >=48, labels['STAI'] <= 33)]]['STAI'].map(lambda x: 'high' if x >= 48 else 'low')
        l = l.to_dict()
    elif label_type == 'stai-3030':
        l = labels[[a or b for a,b in zip(labels['STAI'] >=44, labels['STAI'] <= 30)]]['STAI'].map(lambda x: 'high' if x >= 44 else 'low')
        l = l.to_dict()
    
    d = pcg_data.copy()
    for k in pcg_data.keys():
        if k not in l.keys():
            del d[k]

    d = resample_and_normalize(d, from_freq=22050, to_freq=to_freq)
    return d, l