#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
import numpy as np 
import time 

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *

import json 

from data import WeightedImageDataset, WeightedDataset, get_dataset, get_data_aug, __DATA_AUG__
from train import train, test 


parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--num_epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--lr_schedule', default=1, type=int, help='lr schedule type')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0., type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100/fmnist]')
parser.add_argument('--method', default='deterministic', type=str, help='method choose between [deterministic, mixup, mpmixup]')
parser.add_argument('--mixupbatch', action='store_true', default=False, help='whether to sample batch beta samples in standard mixup')
parser.add_argument('--npl_weight', default=1.0, type=float, help='weight for OOD augmentation data')
parser.add_argument('--npl_prop', default=1.0, type=float, help='proportion of OOD augmentation data')
parser.add_argument('--alpha', default=1.0, type=float, help='mixup alpha')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--output_dir', type=str, default='../outputs', help='output directory')
parser.add_argument('--quicktest', action='store_true', help='quick test running the code')

# save frequency 
parser.add_argument('--save_every', type=int, default=50, help='save model and test prediction frequency.')

# load saved model checkpoint
parser.add_argument('--saved_model_dir', type=str, default='', help='Directory for saved model.')
parser.add_argument('--saved_model_name', type=str, default='', help='Saved model name')
parser.add_argument('--saved_epoch', type=int, default=0, help='Last trained epoch for the saved models')

# test
parser.add_argument('--testdata', type=str, default='', help='A single test dataset, or a string of test datasets connected by -')
parser.add_argument('--test_savefolder', type=str, default='', help='save folder name for test prediction')
parser.add_argument('--test_num_classes', type=int, default=10)

args = parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device 


# Return network & file name
def getNetwork(args, num_classes):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    elif args.net_type == 'FFNN':
        activation = nn.ReLU() 
        hidden_layers= [200, 200] #[100, 100] 
        try:
            input_dim = uci_data_dim_map[args.dataset]
        except Exception as e:
            raise NotImplementedError(f'dataset {args.dataset} not implemented for FFNN.\
                Supported datasets = {uci_data_dim_map.keys()}')
        if num_classes == 2:
            # binary prediction 
            net = FFNN(input_dim=input_dim, output_dim=1, activation=activation, \
                hidden_layers=hidden_layers)
        else:
            net = FFNN(input_dim=input_dim, output_dim=num_classes, activation=activation, \
                hidden_layers=hidden_layers)
        file_name = 'FFNN'
    elif args.net_type == 'CNN':
        # for MNIST or FashionMNIST
        net = CNN(input_dim=28, input_channels=1, num_classes=num_classes)
        file_name = 'CNN'
    elif args.net_type == 'resnet18':
        # for FashionMNIST
        net = resnet18(pretrained=False, num_classes=num_classes)
        file_name = 'resnet18'
    else:
        raise ValueError(f'Got {args.net_type} Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet / CNN / Resnet18')
        sys.exit(0)

    return net, file_name
       
            
def main():
    '''
    loads data, trains model, evaluates on test set.
    Saves metrics and model every args.save_every epochs
    '''
    if args.method == 'mixup':
        method_name = f'{args.method}-batch{args.mixupbatch}-{args.alpha}'
    elif args.method in __DATA_AUG__.keys() or args.method == 'mpmixup':
        method_name = f'{args.method}-{args.alpha}-w{args.npl_weight}-p{args.npl_prop}'
    else:
        method_name = args.method 
    args.output_dir = os.path.join(args.output_dir, args.dataset, \
                                   f"{method_name}_s{args.lr_schedule}_bsz{args.batch_size}_wd{args.weight_decay}_seed{args.seed}_" \
                                   + time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    # initialize the model, or load
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print(f"\nParameters")
    print(json.dumps(args.__dict__, indent=4))
    print("\n")

    with open(os.path.join(args.output_dir, "params.json"), "w") as json_file:
        json.dump(args.__dict__, json_file, indent=4)
        

    # Hyper Parameter settings
    device = get_device()
    print("\nDevice = ", device)
    
    num_epochs = args.num_epochs 
    batch_size = args.batch_size 
    
    if args.quicktest:
        num_epochs = 1 

    # Data 
    print('\n[Phase 1] : Data Preparation')
    
    if args.dataset in ['cifar10', 'cifar100']:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
        ])
    elif args.dataset == 'fmnist':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    
    # load training data
    if args.dataset in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
        ]) # meanstd transformation

        if(args.dataset == 'cifar10'):
            print("| Preparing CIFAR-10 dataset...")
            sys.stdout.write("| ")
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
            num_classes = 10
        elif(args.dataset == 'cifar100'):
            print("| Preparing CIFAR-100 dataset...")
            sys.stdout.write("| ")
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
            num_classes = 100
            
        train_x, train_y = trainset.data, trainset.targets 
        
    elif args.dataset == 'fmnist':
        print("| Preparing FMNIST dataset...")
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])        
        trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)
            
        num_classes = 10
        train_x, train_y = trainset.data.numpy(), trainset.targets.numpy()
            
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")


    weights = torch.ones(train_x.shape[0]) 
    
    if args.dataset in ['cifar10', 'cifar100', 'fmnist']:
        trainset = WeightedImageDataset(data=train_x, labels=train_y, weights=weights, transform=transform_train) 
    else:
        trainset = WeightedDataset(data=train_x, labels=train_y, weights=weights)
            
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=4)
    
    
    # Model
    print('\n[Phase 2] : Model setup')
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args, num_classes)
    if file_name != 'resnet18':
        net.apply(conv_init)
     
    net = net.to(device)
    
    if args.saved_model_dir != '': 
        net.load_state_dict(torch.load(os.path.join(args.saved_model_dir, args.saved_model_name), \
                                    map_location=device))
            
        print("Successfully loaded model")
        start_epoch = args.saved_epoch + 1 
    else:
        start_epoch = 1 

    if net.output_dim == 1:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
        
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    
    
    if args.dataset == 'cifar10':
        lr_scheduler = cf.learning_rate 
    elif args.dataset == 'cifar100':
        if args.lr_schedule == 1:
            lr_scheduler = cf.learning_rate 
        elif args.lr_schedule == 2:
            lr_scheduler = cf.learning_rate2
    elif args.dataset == 'fmnist':
        lr_scheduler = cf.learning_rate
    else:
        lr_scheduler = lambda _lr, _epoch: _lr 
    
    current_lr = lr_scheduler(args.lr, 0)
    if args.dataset in ['cifar10', 'cifar100', 'fmnist']:
        optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.dataset in uci_data_dim_map.keys():
        optimizer = optim.Adam(net.parameters(), lr=current_lr, weight_decay=args.weight_decay)

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch+num_epochs): 
        start_time = time.time()
        
        # adjust learning rate
        current_lr = lr_scheduler(args.lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
         
        train_acc, train_loss = train(epoch,  trainloader, net, optimizer, criterion, \
                                      device, args.method, args.alpha, args.npl_weight, \
                                      args.npl_prop, args.quicktest, current_lr, \
                                      mixupbatch=args.mixupbatch)
        test_acc, test_loss, test_logits = test(epoch, testloader, net, criterion, device, \
            quicktest=args.quicktest)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Epoch time : %d:%02d:%02d'  %(cf.get_hms(epoch_time)))
        print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
        
        if  epoch != start_epoch and ( (epoch - start_epoch + 1) % args.save_every == 0 or epoch == start_epoch+num_epochs-1):
            print(f"| Save predictions and model in epoch {epoch}.")
            
            # save test model and predictions 
            np.savez_compressed(os.path.join(args.output_dir, f'pred_epoch{epoch}.npz'), 
                                test_acc=test_acc, test_loss=test_loss, test_logits=test_logits,
                                train_acc=train_acc, train_loss=train_loss)
            
            model_savename = os.path.join(args.output_dir, f'model_epoch{epoch}.pt')
            torch.save(net.state_dict(), model_savename)
    
        
    print(f"test_acc =  {test_acc:.2f}%")
    print(f"test_loss = {test_loss:.2f}")
    print(f"train_acc = {train_acc:.2f}%")
    print(f"train_loss = {train_loss:.2f}")
    
    print("Finish experiment.")


if __name__ == "__main__":
    main()

