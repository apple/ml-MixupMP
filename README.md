# Posterior Uncertainty Quantification in Neural Networks using Data Augmentation

This software project accompanies the research paper, [Posterior Uncertainty Quantification in Neural Networks using Data Augmentation](https://arxiv.org/abs/2403.12729).


In particular, this repository contains code to implement MixupMP, a method for sampling from the Martingale Posterior of a neural network given a mixup-based predictive distribution.

## Documentation

This repository contains code to implement Deep Ensembles, Mixup Ensembles, and MixupMP, using the datasets and architectures described in the above paper. 



## Getting Started

Install dependencies by running `requirements.txt`:

```
pip install -r requirements.txt
```

The `scripts/` folder contains a script to train a single ensemble member, providing the appropriate parameters for `main.py`.

```
cd scripts
./cifar10_train.sh
```
 
This script can be modified to use different methods or datasets. To create an ensemble, modify this scripts to use different seeds.

Tools for evaluating individual runs and ensembles can be found in `utils.py`.
