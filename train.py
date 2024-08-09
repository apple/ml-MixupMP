#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import torch 
from torch.distributions.beta import Beta
import numpy as np 
import sys 

from data import get_data_aug, __DATA_AUG__
from utils import compute_ece


def mixup_data(x, y, alpha=1.0, batch_lam=False):
    '''Implements mixup
    Inputs:
    x: torch tensor of data input
    y: torch tensor of data targets
    alpha: float indicating mixup parameter, lam ~ Beta(alpha, alpha). If alpha<=0, use deterministic lam = 1
    batch_lam: bool. If true, sample separate lam for all batch members; otherwise, sample single lam per batch

    Returns:
    mixed_x: torch tensor containing mixed input
    y_a, y_b: torch tensor containing pairs of ys; can construct mixed_y from y_a, y_b and lam
    lam: lambda values used for mixup

    '''
    if alpha > 0:
        if batch_lam:
            lam = Beta(alpha,alpha).sample((x.shape[0],)).to(x.device)
        else:
            lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    if batch_lam:
        for _ in range(len(x.shape) -1):
            lam = lam.unsqueeze(-1)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        lam = lam.squeeze()
    else:
        mixed_x = lam * x + (1 - lam) * x[index, :]
   
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    Applies loss criterion to mixup data
    Inputs:
    criterion: torch criterion for loss evaluation
    pred: predicted value
    y_a, y_b: torch tensor containing pairs of ys; can construct mixed_y from y_a, y_b and lam
    lam: lambda values used for mixup

    Returns:
    loss criterion evaluated on mixup data
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training
def train(epoch, trainloader, net, optimizer, criterion, device, 
          method, alpha=1.0, 
          npl_weight=0.1, npl_prop=1., 
          quicktest=False, current_lr=None, 
          mixupbatch=False):
    '''
    Runs one epoch of training
    Inputs:
    epoch: int, epoch number (only used for printed output)
    trainloader: torch.utils.data.DataLoader wrapped around a WeightedImageDataset or WeightedDataset
    net: neural network
    optimizer: torch optimizer
    criterion: torch criterion
    device: device on which net lives
    method: 'mixup', 'mpmixup', or 'deterministic'
    alpha: float, mixup parameter
    npl_weight: float, ratio r in paper (assuming npl_prop = 1.)
    npl_prop: float, ratio of number of observations: number of mixup observations
    quicktest: bool; if true, only look at one batch
    current_lr: float; current learning rate (only for printing output)
    mixupbatch: bool; If true, sample separate lam for all batch members; otherwise, sample single lam per batch

    Returns:
    train_acc: average train accuracy
    train_loss: train loss
    '''
    
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    
    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, current_lr))
    
    for batch_idx, (inputs, targets, weights) in enumerate(trainloader):
        if quicktest and batch_idx == 1:
            break 
        
        inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device) 
        if method == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, batch_lam=mixupbatch)
       
        elif method == 'mpmixup':
            n_original = len(inputs)
            n_pseudo = int(n_original * npl_prop)
            pseudo_indices = torch.randperm(n_original)[:n_pseudo]
            weights_pseudo = torch.ones(n_pseudo, device=device) * npl_weight
            weights = torch.cat([weights, weights_pseudo], dim=0)
            inputs_pseudo, targets_pseudo_a, targets_pseudo_b, lam = mixup_data(inputs[pseudo_indices], targets[pseudo_indices], alpha, \
                                                                                batch_lam=True)
            inputs = torch.cat([inputs, inputs_pseudo], dim=0)         
        
        optimizer.zero_grad()
        outputs = net(inputs)               # Forward Propagation
        
        if method == 'mixup':
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif method == 'mpmixup':
            pseudo_loss = mixup_criterion(criterion, outputs[n_original:], targets_pseudo_a, targets_pseudo_b, lam)
            original_loss = criterion(outputs[:n_original], targets)
            loss = torch.cat([original_loss, pseudo_loss], dim=0)
        else:
            loss = criterion(outputs, targets)  # Loss

        if (method in __DATA_AUG__.keys()):
            original_loss = torch.mean(loss.detach()[:n_original], dim=0).item() 
        else:
            original_loss = None 
        
        loss = torch.mean(weights * loss, dim=0)  
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        if len(outputs.shape) == 2:      
            _, predicted = torch.max(outputs.data, 1)
        else:
            predicted = (outputs.data > 0)
        total += targets.size(0)
        
        if method == 'mixup':
            correct += ((lam * predicted.eq(targets_a.data)).cpu().sum().float() 
                        + ((1 - lam) * predicted.eq(targets_b.data)).cpu().sum().float())
        elif method == 'mpmixup':
            predicted_original = predicted[:n_original]
            predicted_pseudo = predicted[n_original:]
            correct_original = predicted_original.eq(targets.data).cpu().sum()
            correct_pseudo = (lam * predicted_pseudo.eq(targets_pseudo_a.data)).cpu().sum().float() \
                + ((1 - lam) * predicted_pseudo.eq(targets_pseudo_b.data)).cpu().sum().float()
            correct += correct_original + correct_pseudo
            total += targets_pseudo_a.size(0)
        else:
            correct += predicted.eq(targets.data).cpu().sum()
        
        train_acc = (100.*correct/total).item()

    epoch_metric_dict = {"epoch_train_loss": train_loss, "epoch_train_acc": train_acc}
    epoch_str_output = f'\n| Training Epoch {epoch}'
    for metric, value in epoch_metric_dict.items():
        epoch_str_output += f' {metric}: {value:.3f}'
    print(epoch_str_output)
        
    return train_acc, train_loss    


def test(epoch, testloader, net, criterion, device, quicktest=False):
    '''
    Evaluation on test set
    Inputs:
    epoch: int, epoch number (only used for printed output)
    testloader: torch.utils.data.DataLoader wrapped around a WeightedImageDataset or WeightedDataset
    net: neural network
    criterion: torch criterion
    device: device on which net lives
    quicktest: bool; if true, only look at one batch

    Returns:
    acc: average test accuracy
    test_loss: test loss
    test_logits: torch tensor containing test logits
    '''
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    
    test_logits = [] 
    test_labels = []
    
    binary = (net.output_dim == 1)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if quicktest and batch_idx == 1:
                break 
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            test_logits.append(outputs.detach().cpu().numpy())
            
            loss = criterion(outputs, targets).mean()

            test_loss += loss.item()
            if not binary:
                _, predicted = torch.max(outputs.data, 1)
            else:
                predicted = (outputs.data > 0)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            
            test_labels.append(targets.cpu().numpy())  


    acc = (100.*correct/total).item() 
    
    test_logits = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    accuracy_list, confidence_list, ece, bin_freq_list, oe, ue = \
        compute_ece(logits=test_logits, labels=test_labels, num_bins=10, verbose=False, binary=binary)
    
    test_metric_dict = {"test_loss": loss.item(), 
                        "test_acc": acc, 
                        "test_ece": ece, 
                        "test_oe": oe, 
                        "test_ue": ue}
    valid_str_output = f'\n| Validation Epoch {epoch}'
    for metric, value in test_metric_dict.items():
        valid_str_output += f' {metric}: {value:.3f}'
    print(valid_str_output)

        
    return acc, test_loss, test_logits 
     
     
