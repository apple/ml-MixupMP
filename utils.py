#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np 
import torch 


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def softmax(logp, axis=-1):
    # logp is logits
    logp_max = np.max(logp, axis=axis, keepdims=True)
    logp = logp - logp_max 
    log_normalizer = np.log(np.sum(np.exp(logp), axis=axis, keepdims=True))
    logp_normalized = logp - log_normalizer
    return np.exp(logp_normalized)
    

def compute_ece(probs=None, logits=None, labels=None, num_bins=10, verbose=False, binary=False):
    ''' given either probs or logits, plus test labels, returns evaluation metrics
    Inputs:
    probs: torch tensor containing class probabilities. If None, logits must be defined.
    logits: torch tensor containing class probabilities. Ignored unless probs is None
    labels: torch tensor containing integer class labels
    num_bins: number of bins for ECE. In the paper, we use 15
    verbose: If True, print out metrics
    binary: bool, set to True for binary classification

    Returns:
    accuracy_list: list of accuracies within each bin. For plotting calibration plots
    confidence_list: list of confidences within each bin. For plotting calibration plots
    ece: Expected Calibration Error
    bin_freq_list: Histogram of counts for each bin
    oe: Overconfidence error; see paper for definition
    ue: Underconfidence error; see paper for definition
    '''
    if probs is None:
        assert logits is not None 
        if binary:
            probs = sigmoid(logits)
        else:
            probs = softmax(logits, axis=-1)

    if binary:
        # probs: (num_data,)
        probs = np.stack([1-probs, probs], axis=1)
        
    # probs: (num_data, num_classes)
    predictions = np.argmax(probs, axis=1)
    
    # labels: (num_classes,)
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy().astype(predictions.dtype)
    
    confidences = np.max(probs, axis=1)
    bin_boundaries = np.linspace(0, 1, num_bins+1)
    ece = 0.0
    oe = 0.0 # over-confidence error: max(conf - acc, 0)
    ue = 0.0 # under-confidence error: max(acc - conf, 0)
    accuracy_list = []
    confidence_list = [] 
    bin_count_list = []
    for bin_idx in range(num_bins):
        if bin_idx == num_bins - 1:
            bin_mask = np.logical_and(confidences >= bin_boundaries[bin_idx], confidences <= bin_boundaries[bin_idx+1])
        else:
                bin_mask = np.logical_and(confidences >= bin_boundaries[bin_idx], confidences < bin_boundaries[bin_idx+1])
        bin_labels = labels[bin_mask]
        bin_predictions = predictions[bin_mask]
        bin_confidences = confidences[bin_mask]
        
        bin_count = len(bin_labels)
        if bin_count > 0:
            accuracy = np.mean(bin_labels == bin_predictions)
            confidence = np.mean(bin_confidences)
            ece += np.abs(accuracy - confidence) * len(bin_labels)
            
            oe += np.maximum(confidence - accuracy, 0) * len(bin_labels)
            ue += np.maximum(accuracy - confidence, 0) * len(bin_labels)
            
            accuracy_list.append(accuracy)
            confidence_list.append(confidence)
            bin_count_list.append(bin_count)
            if verbose:
                print("\nbin_idx", bin_idx)
                print("confidence", confidence)
                print("accuracy", accuracy)
                print("bin_count", bin_count)
            
    ece /= np.sum(labels.shape)
    oe /= np.sum(labels.shape)
    ue /= np.sum(labels.shape)
    
    bin_count_list = np.array(bin_count_list, dtype=np.float64)
    bin_freq_list = bin_count_list / np.sum(bin_count_list)
    return accuracy_list, confidence_list, ece, bin_freq_list, oe, ue 
