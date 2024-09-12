# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models

from transopt.benchmark.HPO import networks

ALGORITHMS = [
    'ERM',
    'ROBERM',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            # momentum=self.hparams['momentum'],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

        
        

class ROBERM(Algorithm):
    """
    Empirical Risk Minimization (ERM) with an additional decoder for reconstruction.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ROBERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        # Featurizer extracts features from the input
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # Classifier performs classification based on the extracted features
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        # Decoder reconstructs the input image from the features
        self.decoder = networks.Decoder(self.featurizer.n_outputs, input_shape)

        # Combining featurizer and classifier for the classification task
        self.network = nn.Sequential(self.featurizer, self.classifier)

        # Define separate optimizers for the classifier and the decoder
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.decoder.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # Classification loss
        features = self.featurizer(all_x)
        classification_loss = F.cross_entropy(self.classifier(features), all_y)

        # Reconstruction loss - the decoder tries to reconstruct the input
        reconstructed_x = self.decoder(features)
        reconstruction_loss = 10 * F.mse_loss(reconstructed_x, all_x)

        # Total loss as the sum of classification and reconstruction losses
        total_loss = classification_loss + reconstruction_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {'classification_loss': classification_loss.item(), 'reconstruction_loss': reconstruction_loss.item()}

    def predict(self, x):
        # Extract features from input
        features = self.featurizer(x)
        # Get the classification output
        labels = self.classifier(features)
        # Get the reconstructed image from the decoder
        reconstructed_x = self.decoder(features)
        # Return both the classification label and the reconstructed image
        return labels, reconstructed_x