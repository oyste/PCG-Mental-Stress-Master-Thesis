import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from kymatio import Scattering1D


def asynchronous_segmentation(data_dict, sfreq, seconds):
    """
    This function segments data into epochs of a specified length and returns a dictionary of the
    segmented data.
    
    :param data_dict: a dictionary containing the data for each recording
    :param sfreq: sampling frequency of the data
    :param seconds: The length of each segment in seconds
    :return: a dictionary containing segmented data from the input data dictionary. The keys of the
    returned dictionary are in the format of "{original_key}_epoch{epoch_number}" and the values are the
    segmented data arrays.
    """
    x_dict = {}
    i = 0
    for k, rec in data_dict.items():
        L = rec.shape[0]
        win_size = int(np.floor(sfreq*seconds))
        N = int(np.floor(L/win_size))
        for win in range(N):
            x_dict[f'{k}_epoch{win}'] = data_dict[k][win*win_size:(win+1)*win_size]
            i += 1
    return x_dict

def y_seg(all_seg, stress_labels):
    """
    The function maps stress labels to segments based on their keys.
    
    :param all_seg: It is a dictionary containing segment names as keys and their corresponding values
    as values. These segments are typically extracted from a larger audio file and are used for further
    analysis or processing
    :param stress_labels: It is a dictionary containing stress labels for each segment. The keys of the
    dictionary are segment IDs and the values are the corresponding stress labels
    :return: The function `y_seg` returns a dictionary `o` where the keys are the same as the keys in
    the input dictionary `all_seg`, and the values are the corresponding stress labels obtained from the
    `stress_labels` dictionary.
    """
    labels = list(stress_labels['_'.join(k.split('_')[:-1])] for k in all_seg.keys())
    o = {}
    for i, e in enumerate(all_seg.keys()):
        o[e] = labels[i]
    return o

def segment_data_and_labels(d, l, method='async', seconds=0.5, sr=1000):
    """
    This function segments data and labels using either asynchronous method with
    specified parameters.
    
    :param d: The input data to be segmented
    :param l: The labels or target values for the data
    :param method: The method used for segmenting the data. Only 'async' supported, defaults
    to async (optional)
    :param seconds: The length of each segment in seconds
    :param sr: sampling rate of the data (in Hz), defaults to 1000 (optional)
    :return: two variables: `d_s` and `l_s`, segmented data and labels.
    """
    if method == 'async':
        d_s = asynchronous_segmentation(d, sfreq=sr, seconds=seconds)
    l_s = y_seg(d_s, l)
    return d_s, l_s

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          title=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=13)
    plt.yticks(tick_marks, classes, fontsize=13)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=15)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)
    plt.show()
    

def full_multiclass_report(model, x, y_true, classes,batch_size=32, binary=False):
    """
    This function generates a full multiclass classification report including accuracy, classification
    report, confusion matrix, and plot for a given model, input data, true labels, and classes.
    
    :param model: The trained machine learning model that you want to evaluate
    :param x: The input data for the model
    :param y_true: The true labels of the data
    :param classes: A list of class names in the order of their indices in the output of the model. For
    example, if the model output is a one-hot encoded vector of length 3, where the first index
    represents class A, the second index represents class B, and the third index represents class C,
    then
    :param batch_size: The batch size is the number of samples that will be propagated through the
    neural network at once during training. It is a hyperparameter that can be tuned to optimize the
    training process, defaults to 32 (optional)
    :param binary: A boolean parameter that specifies whether the problem is binary or multiclass. If
    binary is True, the function assumes that the problem is binary classification and converts the true
    labels to binary format. If binary is False, the function assumes that the problem is multiclass
    classification and uses the true labels as they are, defaults to False (optional)
    """

    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    y_proba = model.predict(x, batch_size = batch_size) 
    y_pred = y_proba > 0.5
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5,labels=[0,1], target_names=classes))    
    
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)

def full_multiclass_report_sklearn(y_true, y_pred, classes, title=None):
    """
    This function prints the accuracy, classification report, and confusion matrix for a multiclass
    classification problem for scikit-learn classifiers.
    
    :param y_true: The true labels of the data, i.e. the correct classification of each sample
    :param y_pred: The predicted labels for the classification task
    :param classes: A list of class labels in the order they appear in the classification report and
    confusion matrix
    """
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5,labels=[0,1], target_names=classes))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    print('specificity: ', cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    print('sensitivity: ', cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1]))
    print('AUC: ', roc_auc_score(y_true, y_pred))
    plot_confusion_matrix(cnf_matrix,classes=classes, title=title)


def score_subjects(subject_ids, y_epochs):
    """
    The function takes in a list of subject IDs and a dictionary of stress levels for each epoch, and
    returns a list of scores for each subject based on their stress levels. It is reffered to as the 
    'Stress Heuristic' in the thesis.
    
    :param subject_ids: A list of subject IDs (strings) for which we want to calculate scores
    :param y_epochs: It is a dictionary where the keys are strings representing the time epochs and the
    values are strings representing the stress level of a subject during that epoch. The stress level
    can be either 'low' or 'high'
    :return: a list of scores for each subject based on their stress levels in the y_epochs dictionary.
    The scores are either 1, -1, or 0, indicating high stress, low stress, or neutral stress
    respectively. The function also ensures that there are no more than two subjects with a neutral
    stress score, and if there are, it assigns them a high or low stress score based
    """
    scores = []
    tres = len(y_epochs)/3/len(subject_ids)
    map = {'low':-1, 'high':1}
    for i,e in enumerate(subject_ids):
        scores.append(np.sum([map[stress] for k, stress in y_epochs.items() if k.split('_')[0]==e]))    
        if scores[i] > tres: scores[i] = 1
        elif scores[i] < -tres: scores[i] = -1
        else: scores[i] = 0
    cnt_zero, count_one, count_n_one = scores.count(0), scores.count(1), scores.count(-1)
    if cnt_zero < 2:
        if count_one > count_n_one:
            scores[scores.index(0)] = -1
        else:
            scores[scores.index(0)] = 1
    return scores


def sd_train_test_and_kfold_split(x_epochs, y_epochs, test_size=0.3, random_state=42, n_splits=5, shuffle=True):
    """
    This function splits data into subject dependent training, validation, and test sets using stratified k-fold
    cross-validation.
    
    :param x_epochs: A dictionary containing the input data for each epoch, where the keys are strings
    representing the subject ID and epoch number (e.g. 'subject1_epoch1') and the values are numpy
    arrays of shape (channels, samples)
    :param y_epochs: The target variable for the epochs data. It could be a list or array of labels or
    scores for each epoch
    :param test_size: The proportion of the dataset to include in the test split
    :param random_state: random_state is a parameter used to initialize the random number generator. It
    is used to ensure that the same random sequence is generated every time the function is run with the
    same input parameters. This is useful for reproducibility and debugging purposes, defaults to 42
    (optional)
    :param n_splits: The number of folds to be used in the cross-validation process, defaults to 5
    (optional)
    :param shuffle: A boolean parameter that determines whether or not to shuffle the data before
    splitting it into folds. If set to True, the data will be shuffled randomly before splitting. If set
    to False, the data will be split in the order it appears, defaults to True (optional)
    :return: four lists: train_keys, val_keys, train_keys_concat, and test_keys.
    """
    subject_ids = np.unique([k.split('_')[0] for k in x_epochs.keys()])
    subject_scores = score_subjects(subject_ids, y_epochs)

    cv_subjects, test_subjects = \
        train_test_split(subject_ids, test_size=test_size, random_state=random_state, shuffle=True, stratify=subject_scores)
    #cv_scores = [subject_scores[i] for i,e in enumerate(subject_ids) if e in cv_subjects]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    train_keys, val_keys = [], []
    
    #for train_i_subjects, val_i_subjects in skf.split(cv_subjects, cv_scores):
    #    train_i_subjects_list = cv_subjects[train_i_subjects]
    #    val_i_subjects_list = cv_subjects[val_i_subjects]
    #    keys = list(x_epochs.keys())
    #    random.shuffle(keys, random=random.seed(random_state))
    #    train_keys.append([k for k in keys if k.split('_')[0] in train_i_subjects_list])
    #    val_keys.append([k for k in keys if k.split('_')[0] in val_i_subjects_list])
    
    for train_i_subjects, val_i_subjects in skf.split(subject_ids, subject_scores):
        train_i_subjects_list = subject_ids[train_i_subjects]
        val_i_subjects_list = subject_ids[val_i_subjects]
        keys = list(x_epochs.keys())
        random.shuffle(keys, random=random.seed(random_state))
        train_keys.append([k for k in keys if k.split('_')[0] in train_i_subjects_list])
        val_keys.append([k for k in keys if k.split('_')[0] in val_i_subjects_list])

    test_keys = [k for k in keys if k.split('_')[0] in test_subjects]
    train_keys_concat = list(i for sub in val_keys for i in sub)

    return train_keys, val_keys, train_keys_concat, test_keys

def prep_data_for_scattering(x, T):
    """
    This function prepares data for the wavelet scattering transform by zero-padding or truncating the input data to a
    specified length.
    
    :param x: The input data to be prepared for scattering. It is expected to be a numpy array
    :param T: T is the length of the time series data that we want to prepare for scattering. The
    function takes in a set of time series data x and prepares it for scattering by zero-padding or
    truncating the data to fit the length T. The resulting data is returned as a numpy array with
    dimensions (number
    :return: The function `prep_data_for_scattering` returns a numpy array `x_all` with shape `(len(x),
    T)`. This array is created by iterating over each element `v` in `x`, reshaping it to a 1D array,
    truncating it if it is longer than `T`, and zero-padding it if it is shorter than `T`. The resulting
    1D
    """
    x_all = np.zeros((len(x), T))
    for i,v in enumerate(x):
        v = v.ravel()
        # If it's too long, truncate it.
        if len(v) > T:
            v = v[:T]
        # If it's too short, zero-pad it.
        start = (T - len(v)) // 2
        x_all[i,start:start + len(v)] = v.reshape(1,-1)
    return x_all

def format_labels_for_tf(train_labels, test_labels):
    """
    This function converts the labels in train_labels and test_labels from string format to integer
    format using a dictionary.
    
    :param train_labels: The labels for the training data, which are typically the target values that
    the machine learning model is trying to predict
    :param test_labels: The test labels are the labels (or target values) for the test dataset. They are
    used to evaluate the performance of a machine learning model on unseen data
    :return: The function `format_labels_for_tf` is returning a tuple of two numpy arrays. The first
    array contains the integer labels corresponding to the `train_labels` input, and the second array
    contains the integer labels corresponding to the `test_labels` input. The integer labels are
    generated by mapping the string labels 'low' and 'high' to the integers 0 and 1, respectively.
    """
    labels_to_int = {
        'low' : 0,
        'high' : 1
    }
    return np.vectorize(labels_to_int.get)(train_labels), np.vectorize(labels_to_int.get)(test_labels)

def make_parent_daughter_index_pairs(meta):
    """
    This function creates index pairs for parent-daughter relationships with first and second order scattering coefficients.
    To be used in the normalized scattering transform, specifically second order coefficient normalization.
    :param meta: The meta information of the scattering class, i.e Scattering1D().meta() 
    :return: The function `make_parent_daughter_index_pairs` returns a list of pairs of indices, where
    each pair represents a parent-daughter relationship between first and second order scattering coefficients. 
    """
    ks = meta['key']
    pairs = []
    parents = []
    for i, k in enumerate(ks):
        if len(k) == 1:
            parents.append([i, k[0]])
    for i, k in enumerate(ks):
        if len(k) == 2:
            for ip, parent in enumerate(parents):
                if parent[1] == k[0]:
                    pairs.append([ip+1, i])
    return pairs

def normalize_scattering_data(Sx, meta, x):
    """
    This function normalizes scattering data by dividing first order coefficient by the average absolute value of x over L
    and each second order coefficient by the value of its first order parent.
    
    :param Sx: a numpy array containing the scattered data from temporal scattering
    :param meta: A dictionary containing metadata about the scattering 
    :param x: the original PCG signal
    :return: The function `normalize_scattering_data` returns the normalized scattering data `Sx_norm`.
    """
    pairs = make_parent_daughter_index_pairs(meta)
    Sx_norm = np.zeros_like(Sx)
    S1_max = len(Sx[0,meta['order'] == 1])+1
    Sx_norm = Sx
    x_abs_avg = np.average(abs(x), axis=1) # average abs(x) over time (L)
    for rec in range(Sx.shape[0]):
        for i in range(1,S1_max):
            for j in range(Sx.shape[2]):
                Sx_norm[rec,i,j] = Sx[rec,i,j]/x_abs_avg[rec] #S_1 normalization
        for parent, daughter in pairs:
            for j in range(Sx.shape[2]):
                Sx_norm[rec,daughter,j] = Sx[rec,daughter,j]/Sx[rec,parent,j] #S_2 normalization
    return Sx_norm

def scatter_x(x, J, Q, normalize=False):
    """
    This function computes the scattering transform of a given input signal and optionally normalizes
    the output.
    
    :param x: a 2D numpy array of shape (batch_size, signal_length) containing the input signals to be
    transformed by the scattering transform
    :param J: The number of scales used in the scattering transform. It determines the invariance scale
    of the network
    :param Q: The Q parameter in the scatter_x function is a positive integer that determines the number
    of logarithmically spaced wavelets used in the scattering transform. It controls the frequency
    resolution of the scattering transform
    :param normalize: A boolean parameter that determines whether the scattering coefficients should be
    normalized or not. If set to True, the scattering coefficients will be normalized using the
    `normalize_scattering_data` function before being returned. If set to False, the scattering
    coefficients will be returned as is, defaults to False (optional)
    :return: The function `scatter_x` returns the scattering coefficients `Sx` of the input signal `x`
    computed using the Scattering1D transform with parameters `J` and `Q`. If the `normalize` parameter
    is set to `True`, the function also normalizes the scattering coefficients using the
    `normalize_scattering_data` function and returns the normalized scattering coefficients `Sx_norm`.
    """
    S = Scattering1D(J=J, shape=x.shape[1], Q=Q, max_order=2)
    Sx = S(x)
    if normalize:
        Sx_norm = normalize_scattering_data(Sx, S.meta(), x) 
        return Sx_norm
    return Sx

def reshape_for_network(x,y):
    """
    The function reshapes input data by flattening the time dimension of x and reshaping both x, and y 
    for the classifier.
    
    :param x: The input scattering data. It is a 3-dimensional numpy array with shape
    (n_samples, num_features, num_timesteps)
    :param y: The variable y is a numpy array that contains the labels or target values for a dataset.
    The shape of y is (n_samples,), where n_samples is the number of segmented recordings in the dataset. Each
    element in y represents the label or target value for a corresponding sample in the dataset
    :return: The function `reshape_for_network` returns two arrays: `x_network` and `y_network`.
    """
    y_network = np.zeros(y.shape[0]*x.shape[2])
    x_network = np.zeros((x.shape[0]*x.shape[2],x.shape[1]))
    for i in range(x.shape[0]):
        l = y[i]
        for j in range(x.shape[2]):
            y_network[i*x.shape[2]+j] = l
            x_network[i*x.shape[2]+j,:] = x[i,:,j]
    return x_network, y_network


def remove_bad_features(J, Q, L, features, sr, keep_5060100hz=False, thres=7):
    """
    This function removes bad features from a set of features based on the frequency of the underlying 
    wavelet and returns the remaining features along with their corresponding frequency values.
    
    :param J: The scale of the scattering transform
    :param Q: Q is the number of wavelets per octave used in the scattering transform. It determines the
    frequency resolution of the transform
    :param L: The length of the input signal
    :param features: A numpy array of shape (L, num_features) containing the scattering coefficients of
    an audio signal
    :param sr: sampling rate of the audio signal
    :param keep_5060100hz: A boolean parameter that determines whether to keep the bad frequencies of
    50Hz, 60Hz, and 100Hz or not. If set to True, these frequencies will be kept and not considered as
    bad frequencies. If set to False, all the bad frequencies will be removed including these three,
    defaults to False (optional)
    :param thres: The threshold value used to determine if a feature should be removed based on its
    proximity to a bad frequency. If the distance between the feature's center frequency and a bad
    frequency is less than or equal to thres, the feature is considered bad and will be removed,
    defaults to 7 (optional)
    :return: a tuple containing four elements:
    1. A numpy array of new features after removing the bad features
    2. A list of the new features corresponding first order wavelet center frequencies
    3. A list of the new features corresponding second order wavelet center frequencies
    4. A list of the original features corresponding first order wavelet center frequencies
    5. A list of the original features corresponding second order wavelet center frequencies
    """
    bad_freqs = [50, 60, 100, 120, 150, 180, 200, 240,250,300,350,360,400,420,450,480,500]
    if keep_5060100hz: bad_freqs = bad_freqs[3:]
    S = Scattering1D(J=J, shape=L, Q=Q, max_order=2)
    xi1s = S.meta()['xi'][:,0]*sr
    xi2s = S.meta()['xi'][:,1]*sr
    sigma1s = S.meta()['sigma'][:,0]*sr
    bad_feats = []
    new_feats = []
    new_feat_xi1s = []
    new_feat_xi2s = []
    for i in range(len(xi1s)):
        if i==0: bad_feats.append(i)
        for freq in bad_freqs:
            if xi1s[i] < freq+sigma1s[i]/2+thres and xi1s[i] > freq-sigma1s[i]/2-thres:
                bad_feats.append(i)
    for i in range(features.shape[1]):
        if i not in bad_feats:
            new_feats.append(features[:,i].reshape(-1,1))
            new_feat_xi1s.append(xi1s[i])
            new_feat_xi2s.append(xi2s[i])
    return np.hstack(new_feats), new_feat_xi1s, new_feat_xi2s, xi1s, xi2s

def make_scatter_data(J, Q, L, data, labels, sr=1000, normalize=False, feature_reduction=False):
    """
    This function is to be used to construct scatter data for the hold out test.
    
    :param J: The number of scales used in the scattering transform
    :param Q: Q is the number of filters per octave in the scattering transform. It determines the
    frequency resolution of the transform
    :param L: L is the length of the segments of data that will be used for scattering
    :param data: The input data to be processed by the scattering transform
    :param labels: The labels parameter is a list or array containing the target labels for the data. It
    should have the same length as the data parameter
    :param sr: sr stands for "sampling rate" and refers to the number of samples per second in the input
    data, defaults to 1000 (optional)
    :param normalize: A boolean parameter that determines whether to normalize the scattering
    coefficients or not. If set to True, the scattering coefficients will be normalized, defaults to
    False (optional)
    :param feature_reduction: A boolean parameter that determines whether or not to perform feature
    reduction on the scatter data. If set to True, the function will remove bad features from the
    scatter data, defaults to False (optional)
    :return: either the original or reduced feature sets for the training and testing data, along with
    their corresponding labels.
    """
    S = Scattering1D(J=J, shape=L, Q=Q, max_order=2)
    xi1s = S.meta()['xi'][:,0]*sr
    xi2s = S.meta()['xi'][:,1]*sr
    sigma1s = S.meta()['sigma'][:,0]*sr

    d_s, l_s = segment_data_and_labels(data, labels, method='async', seconds=L/sr, sr=sr)

    train_keys, val_keys, train_keys_concat, test_keys = \
    sd_train_test_and_kfold_split(x_epochs=d_s, y_epochs=l_s, test_size=0.3, random_state=42, n_splits=3, shuffle=True)
    print(len(train_keys_concat))
    y_train = np.vstack([v for v in [l_s[k] for k in train_keys_concat]]).ravel()
    y_test = np.vstack([v for v in [l_s[k] for k in test_keys]]).ravel()
    y_train, y_test = format_labels_for_tf(y_train, y_test)
    x_train = prep_data_for_scattering([d_s[k] for k in train_keys_concat], L)
    x_test = prep_data_for_scattering([d_s[k] for k in test_keys], L)

    x_train_scatter = scatter_x(x_train, J, Q, normalize=normalize)
    x_test_scatter = scatter_x(x_test, J, Q, normalize=normalize)

    x_train_network, y_train_network = reshape_for_network(x_train_scatter, y_train)
    x_test_network, y_test_network = reshape_for_network(x_test_scatter, y_test)
    if not feature_reduction:
        return x_train_network, y_train_network, x_test_network, y_test_network
    x_train_network_rm, _, _, _, _ = remove_bad_features(J, Q, L, x_train_network, sr)
    x_test_network_rm, _, _, _, _ = remove_bad_features(J, Q, L, x_test_network, sr)
    return x_train_network_rm, y_train_network, x_test_network_rm, y_test_network

def make_cross_val_scatter_data(J, Q, L, data, labels, n_splits=3, normalize=False, sr=1000, feature_reduction=False, test_size=0.2):
    """
    This function creates data to be used in cross validation.
    
    :param J: The maximum scale of the scattering transform
    :param Q: The number of wavelets per octave used in the scattering transform
    :param L: The length of each segment of data in seconds
    :param data: The EEG data to be used for cross-validation
    :param labels: The labels parameter is a list or array containing the target labels for the data. It
    should have the same length as the data parameter
    :param n_splits: The number of splits to use for cross-validation, defaults to 3 (optional)
    :param normalize: A boolean indicating whether to compute the normalized scattering transform, defaults to
    False (optional)
    :param sr: sampling rate of the data in Hz (samples per second), defaults to 1000 (optional)
    :param feature_reduction: A boolean indicating whether or not to perform feature reduction on the
    scattering transform data. If True, the function will remove bad features from the data, defaults to
    False (optional)
    :return: four lists: x_train_scatter, y_train_scatter, x_val_scatter, and y_val_scatter. These lists
    contain the training and validation data and labels that have been preprocessed using the scattering
    transform and potentially feature reduction techniques.
    """

    d_s, l_s = segment_data_and_labels(data, labels, method='async', seconds=L/sr, sr=sr)

    train_keys, val_keys, _, _ = \
    sd_train_test_and_kfold_split(x_epochs=d_s, y_epochs=l_s, test_size=test_size, random_state=42, n_splits=n_splits, shuffle=True)
    x_train_scatter = []
    y_train_scatter = []
    x_val_scatter = []
    y_val_scatter = []

    tr_val_keys = list(np.unique(np.hstack([np.unique(np.hstack(train_keys)), np.unique(np.hstack(val_keys))])))
    
    # for optimization we compute the scattering transform for all data
    all_data = prep_data_for_scattering([d_s[k] for k in tr_val_keys], L)
    keys_used_idx = [tr_val_keys.index(k) for k in tr_val_keys]
    all_labels = np.vstack([l_s[k] for k in tr_val_keys]).ravel()
    all_labels, _ = format_labels_for_tf(all_labels, all_labels)
    x_scatter = scatter_x(all_data, J, Q, normalize=normalize)

    for i in range(n_splits):
        
        x_train = np.stack([x_scatter[j,:,:] for j in keys_used_idx if tr_val_keys[j] in train_keys[i]], axis=0)
        x_val = np.stack([x_scatter[j,:,:] for j in keys_used_idx if tr_val_keys[j] in val_keys[i]], axis=0)
        y_train = np.vstack([all_labels[j] for j in keys_used_idx if tr_val_keys[j] in train_keys[i]]).ravel()
        y_val = np.vstack([all_labels[j] for j in keys_used_idx if tr_val_keys[j] in val_keys[i]]).ravel()
        
        x_train_scatter_i, y_train_scatter_i = reshape_for_network(x_train, y_train)
        x_val_scatter_i, y_val_scatter_i = reshape_for_network(x_val, y_val)
        
        if not feature_reduction:
            x_train_scatter.append(x_train_scatter_i)
            x_val_scatter.append(x_val_scatter_i)
            y_train_scatter.append(y_train_scatter_i)
            y_val_scatter.append(y_val_scatter_i)
            continue
        
        x_train_scatter_i_rm, _, _, _, _ = remove_bad_features(J, Q, L, x_train_scatter_i, sr)
        x_val_scatter_i_rm, _, _, _, _ = remove_bad_features(J, Q, L, x_val_scatter_i, sr)

        x_train_scatter.append(x_train_scatter_i_rm)
        x_val_scatter.append(x_val_scatter_i_rm)
        y_train_scatter.append(y_train_scatter_i)
        y_val_scatter.append(y_val_scatter_i)

    return x_train_scatter, y_train_scatter, x_val_scatter, y_val_scatter


def L_predictor(model, L, T, x, y):
    """
    The function takes in a model, signal length (L), time interval (T), input data (x), and true labels
    (y), and returns the true labels and predicted labels for each segment of the input data.
    
    :param model: a trained machine learning model
    :param L: The length of the signal segment used for prediction
    :param T: T is a parameter that determines the length of each segment of the input signal x that is
    fed into the model for prediction. Specifically, the input signal x is divided into segments of
    length L, and each segment is further divided into T sub-segments of equal length. The model then
    makes a prediction
    :param x: The input data for the model, which is a numpy array of shape (n_samples, n_features)
    :param y: The parameter "y" is a numpy array containing the true labels for the input data "x"
    :return: two lists: y_true and y_preds. These lists contain the true labels and predicted labels,
    respectively, for each segment of length L in the input signal x.
    """

    y_preds = []
    y_true = []
    for i in range(int(x.shape[0]/int(L/T))):
        sig_LT_T = x[i*int(L/T):(i+1)*int(L/T),:]
        label = y[i*int(L/T)]
        proba = model.predict(sig_LT_T)
        y_pred = np.average(proba)>0.5
        y_true.append(label)
        y_preds.append(y_pred)
    return y_true, y_preds

def L_predictor_proba(model, L, T, x, y):
    y_preds_proba = []
    y_true = []
    for i in range(int(x.shape[0]/int(L/T))):
        sig_LT_T = x[i*int(L/T):(i+1)*int(L/T),:]
        label = y[i*int(L/T)]
        proba = model.predict(sig_LT_T)
        y_true.append(label)
        y_preds_proba.append(np.average(proba))
    return y_true, y_preds_proba

def L_predictor_2(model, L, T, x, y):
    x_T = []
    y_T = []
    for i in range(int(x.shape[0]/int(L/T))):
        x_T.append(x[i*int(L/T):(i+1)*int(L/T),:])
        y_T.append(y[i*int(L/T)])
    x_T = np.vstack(x_T)
    proba = model.predict(x_T)
    y_pred = np.zeros(int(x.shape[0]/int(L/T)))
    for i in range(int(x.shape[0]/int(L/T))):
        y_pred[i] = np.average(proba[i*int(L/T):(i+1)*int(L/T)])>0.5
    return y_T, y_pred.tolist()