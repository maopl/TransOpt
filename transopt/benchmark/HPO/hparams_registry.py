# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np



def get_hparams(algorithm, dataset):
    """
    Returns a dictionary of hyperparameters for the given algorithm and dataset.
    Each hyperparameter is represented by a tuple (default_value, range_min, range_max).
    """
    hparams = {}

    # Common hyperparameters
    # hparams['data_augmentation'] = (True, None, None)  # Boolean, no range
    # hparams['resnet18'] = (False, None, None)  # Boolean, no range
    # hparams['resnet_dropout'] = (0., 0., 0.5)
    # hparams['class_balanced'] = (False, None, None)  # Boolean, no range
    # hparams['nonlinear_classifier'] = (False, None, None)  # Boolean, no range


    # Dataset-specific hyperparameters
    if dataset in ['MNIST','RotatedMNIST', 'ColoredMNIST']:
        hparams['lr'] = (1e-3, 1e-5, 1e-2)
        hparams['weight_decay'] = (0., 0., 0.)
        hparams['batch_size'] = (64, 8, 512)
        hparams['momentum'] = (0.9, 0.0, 1.0)
        hparams['dropout_rate'] = (0.1, 0.0, 0.5)
    else:
        hparams['lr'] = (5e-5, 1e-6, 1e-3)
        hparams['weight_decay'] = (0., 1e-6, 1e-2)
        hparams['batch_size'] = (32, 8, 128)
        hparams['momentum'] = (0.9, 0.0, 1.0)
        hparams['dropout_rate'] = (0.1, 0.0, 0.5)


    # Algorithm-specific hyperparameters
    if algorithm == 'ERM':
        pass  # ERM doesn't have specific hyperparameters
    elif algorithm == 'ROBERM':
        pass
    # Add more algorithms as needed...

    return hparams

