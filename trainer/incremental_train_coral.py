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





    
#? https://arxiv.org/abs/1607.01719  HDDA

def incremental_train_coral(args, model, model_old, src_trainloader, tgt_trainloader, src_testloader , testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):
    T = 2
    beta = 0.25
    the_lambda = 5
    

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    # model_old = model_old.to(device)

    # model_new = model_new.to(device)
    

    for epoch in range(args.epochs):
        # Set the model to the training mode
        model.train()
        model_old.eval()
        # model_new.train()


        # Set all the losses to zeros
        train_loss = 0
        train_loss_cls = 0
        train_loss_coral = 0
        # Set the counters to zeros

        src_correct = 0
        tgt_correct = 0



        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx,((source_data, source_label),(target_data, target_label)) in enumerate(zip(src_trainloader,tgt_trainloader)): 
            source_data, source_label, target_data, target_label = source_data.to(device), source_label.to(device), target_data.to(device), target_label.to(device)

            tg_optimizer.zero_grad()
            # batch_size = len(source_data)
            # inputs = torch.cat((source_data, target_data), 0)
            # outputs, features = model(inputs)
            # old_outputs = outputs[:batch_size]
            # new_outputs = outputs[batch_size:]
            # src_features = features[:batch_size]
            # tgt_features = features[batch_size:]
            old_outputs, src_features = model(source_data)
            new_outputs, tgt_features = model(target_data)

  
            _, src_predicted = old_outputs.max(1)
            _, tgt_predicted = new_outputs.max(1)
     

            # src_outputs_old, src_features = model_old(source_data)
            
            # classification loss
            loss1 = nn.CrossEntropyLoss(weight_per_class)(old_outputs, source_label) 
            # CORAL loss
            loss2 = CORAL(src_features,tgt_features)
            loss = loss1 + loss2
     
            src_correct += src_predicted.eq(source_label).sum().item()
            tgt_correct += tgt_predicted.eq(target_label).sum().item()
      

            total += source_label.size(0)
            # BP
            loss.backward()
            train_loss += loss.item()
            train_loss_cls += loss1.item() 
            train_loss_coral  += loss2.item()
            tg_optimizer.step()

   
        # Learning rate decay
        tg_lr_scheduler.step()

        
        # Print the training losses and accuracies
        print(tg_lr_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f} train loss cls {:.4f} coral loss {:.4f} src_accuracy: {:.4f} tgt_accuracy: {:.4f} '.format(
            len(src_trainloader), train_loss/(batch_idx+1), train_loss_cls/(batch_idx+1), train_loss_coral/(batch_idx+1), 100.*src_correct/total, 100.*tgt_correct/total))

        # Running the test for this epoch
        model.eval()

        
        test_loss = 0
        src_correct = 0
        tgt_correct = 0


        total = 0
        with torch.no_grad():
            for batch_idx,((source_data, source_label), (target_data, target_label)) in enumerate(zip(src_testloader,testloader)): 

                source_data, source_label, target_data, target_label = source_data.to(device), source_label.to(device), target_data.to(device), target_label.to(device)

                
                src_outputs,_ = model(source_data)
                _, src_predicted = src_outputs.max(1)
                

                src_correct += src_predicted.eq(source_label).sum().item()
                
                tgt_outputs,_ = model(target_data)
                _, tgt_predicted = tgt_outputs.max(1)
                loss = nn.CrossEntropyLoss(weight_per_class)(tgt_outputs, target_label)
                test_loss += loss.item()
                total += target_label.size(0)
                tgt_correct += tgt_predicted.eq(target_label).sum().item()
                

                
        print('Test set: {} test loss: {:.4f} src_accuracy: {:.4f} tgt_accuracy: {:.4f}'.format(len(tgt_trainloader), test_loss/(batch_idx+1), 100.*src_correct/total, 100.*tgt_correct/total))

    return model