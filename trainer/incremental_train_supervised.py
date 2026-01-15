#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:44:14 2022

@author: cbj
"""

import torch

import numpy as np
import torch.nn as nn


import torch.nn.functional as F
import torch.optim as optim

import scipy.io as sio 
from scipy.fftpack import fft
from scipy.spatial.distance import cdist
import torch.utils.data as Data




import loss_function
import time




    


def incremental_train(args, model, model_old,  trainloader,  testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):
    T = 2
    beta = 0.25
    the_lambda = 5
    

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    model_old = model_old.to(device)

    # model_new = model_new.to(device)
    

    for epoch in range(args.epochs):
    # for epoch in range(30):
        # Set the 1st branch model to the training mode
        model.train()
        model_old.eval()
        

        # Set all the losses to zeros
        train_loss = 0
        train_loss_contrastive = 0
        train_loss1 = 0
        train_loss2 = 0
        # Set the counters to zeros
        correct = 0
        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')


        for batch_idx, (inputs, labels) in enumerate(trainloader):
            # Get a batch of training samples, transfer them to the device
            # print(index)
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear the gradient of the paramaters for the tg_optimizer

            tg_optimizer.zero_grad()
            # Forward the samples in the deep networks
            outputs, features = model(inputs)


            # Compute classification loss
            loss1 = nn.CrossEntropyLoss(weight_per_class)(outputs, labels)
            # loss2 = mmd.contrastive_loss(features.detach(),labels)
            loss = loss1 
            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()


            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            # train_loss_contrastive += loss2.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Learning rate decay
        tg_lr_scheduler.step()
        # Print the training losses and accuracies
        print(tg_lr_scheduler.get_lr()[0])
        print('Train set: {} train loss: {:.4f} contrastive_loss {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss/(batch_idx+1), train_loss_contrastive/(batch_idx+1), 100.*correct/total))


        # Running the test for this epoch
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        # tem_testloader = testloader
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        # torch.save(model_1, osp.join(self.save_path, 'b1_true.pkl'))
        # wandb.log({"Test Accuracy": 100.*correct/total, "Test Loss": test_loss/(batch_idx+1)})
        # wandb.finish()
    return model