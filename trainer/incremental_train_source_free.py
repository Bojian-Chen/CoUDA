#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:44:14 2022

@author: cbj
"""

import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.optim as optim

from scipy.fftpack import fft
from scipy.spatial.distance import cdist
import torch.utils.data as Data
import matplotlib.pyplot as plt
from loss_function import *
import copy



def incremental_train_source_free(args, model, model_old, trainloader , testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):
    T = 2
    best_acc = 0
    best_model = None
    

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    model_old = model_old.to(device) 


    for epoch in range(args.epochs):
        # Set the model to the training mode
        model.train()
        model_old.eval()
   
        train_loss = 0
        train_loss_cls = 0
        train_loss_entropy = 0
        train_loss_diventropy = 0
        # Set the counters to zeros
        correct = 0


        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
       
            batch_size = inputs.size(0)
            # train model_new
            tg_optimizer.zero_grad()
     
            
            # Forward the samples in the deep networks
            outputs, features = model(inputs)
            outputs_old, features_old = model_old(inputs)

            outputs_softmax = nn.Softmax(dim=1)(outputs)
    

            
            # classification loss
            loss1 = nn.CrossEntropyLoss(weight_per_class)(outputs, outputs_old / T)*0 
            # loss1 = nn.CosineEmbeddingLoss()(features_old, features, torch.ones(batch_size).to(device))*0
            # Entropy Loss
            loss2 = Entropy(outputs_softmax)
            # DivEntropy Loss
            loss3 = DivEntropy(outputs_softmax)

            loss = loss1 + loss2 - loss3

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
      

            # BP
            loss.backward()
            train_loss += loss.item()
            train_loss_cls += loss1.item() 
            train_loss_entropy  += loss2.item()
            train_loss_diventropy  += loss3.item()

            tg_optimizer.step()

   
        # Learning rate decay
        tg_lr_scheduler.step()

        
        # Print the training losses and accuracies
        print(tg_lr_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f} train loss cls {:.4f} train loss Entropy {:.4f} train loss DivEntropy {:.4f} accuracy: {:.4f} '.format(
            len(trainloader), train_loss/(batch_idx+1), train_loss_cls/(batch_idx+1), train_loss_entropy/(batch_idx+1), train_loss_diventropy/(batch_idx+1), 100.*correct/total))

        # Running the test for this epoch
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

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


        if 100.*correct/total > best_acc:
            best_acc = 100.*correct/total
            best_model = copy.deepcopy(model)
    model = best_model

    return model