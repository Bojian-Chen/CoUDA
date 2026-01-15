#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:40:31 2022

@author: cbj
"""

import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
from loss_function import *

"""
        Training Procedures        
"""

def base_train(args, model, trainloader, testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(args.base_epochs):
        # Set the model to the training mode
        model.train()

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
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()
            # Forward the samples in the deep networks
            outputs, features = model(inputs)
            

            outputs_softmax = nn.Softmax(dim=1)(outputs)
            # Compute classification loss
            loss1 = nn.CrossEntropyLoss(weight_per_class)(outputs, labels)
       
            # Compute contrastive loss
            loss2 = Supervised_InfoNCE_loss(features,labels,temperature=args.temperature)* args.weight_contrastive

        
            
            
            if args.contrastive_loss:
                loss = loss1 + loss2
            else:

                loss = loss1  + loss2*0
            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss_contrastive += loss2.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Learning rate decay
        tg_lr_scheduler.step()
        # Print the training losses and accuracies
        print(tg_lr_scheduler.get_last_lr()[0])
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

    return model
