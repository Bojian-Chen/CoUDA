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


def incremental_train(args, model, model_old, src_trainloader, tgt_trainloader, src_testloader , testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):

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


        # Set all the losses to zeros
        train_loss = 0
        train_loss_cls = 0
        train_loss_entropy = 0
        train_loss_mmd = 0
        train_loss_distillation = 0

        # Set the counters to zeros
        src_correct = 0
        tgt_correct = 0

        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx,((source_data, source_label),(target_data, target_label)) in enumerate(zip(src_trainloader,tgt_trainloader)): 
            source_data, source_label, target_data, target_label = source_data.to(device), source_label.to(device), target_data.to(device), target_label.to(device)
            
            batch_size = len(source_data)
            
            # train model_new
            tg_optimizer.zero_grad()
            inputs = torch.cat((source_data, target_data), 0)
            
            # Forward the samples in the deep networks

            new_outputs, new_features = model(inputs)
    
   
            _, src_predicted = new_outputs[ : batch_size].max(1)
            _, tgt_predicted = new_outputs[batch_size : ].max(1)
     

            _, old_features = model_old(source_data)
            
            # classification loss
            loss1 = nn.CrossEntropyLoss(weight_per_class)(new_outputs[ : batch_size], source_label) 
            # Entropy Loss
            outputs_softmax = nn.Softmax(dim=1)(new_outputs[batch_size : ])
            loss2 = Entropy(outputs_softmax)
            # LMMD loss
            loss3 = lmmd(new_features[ : batch_size], new_features[batch_size : ], source_label, F.softmax(new_outputs[batch_size : ], dim=1), args.nb_cl) 
            # Distillation Loss
            loss4 = nn.CosineEmbeddingLoss()(old_features, new_features[ :batch_size], torch.ones(batch_size).to(device))

            belta = epoch / args.epochs

            loss = loss1  + loss2 * belta + loss3 * (1 - belta) + loss4

     
            src_correct += src_predicted.eq(source_label).sum().item()
            tgt_correct += tgt_predicted.eq(target_label).sum().item()
      

            total += source_label.size(0)
            # BP
            loss.backward()
            train_loss += loss.item()
            train_loss_cls += loss1.item() 
            train_loss_entropy += loss2.item()
            train_loss_mmd += loss3.item()
            train_loss_distillation = loss4.item()
            tg_optimizer.step()

   
        # Learning rate decay
        tg_lr_scheduler.step()

        
        # Print the training losses and accuracies
        print(tg_lr_scheduler.get_last_lr()[0])
        print('Train set: {} train loss: {:.4f}  cls {:.4f}  entropy {:.4f}  mmd loss {:.4f}  distillation {:.4f} src_accuracy: {:.4f} tgt_accuracy: {:.4f} '.format(
            len(src_trainloader), 
            train_loss/(batch_idx+1), 
            train_loss_cls/(batch_idx+1), 
            train_loss_entropy/(batch_idx+1),
            train_loss_mmd/(batch_idx+1), 
            train_loss_distillation/(batch_idx+1),
            100.*src_correct/total, 
            100.*tgt_correct/total))

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


        if 100.*tgt_correct/total >= best_acc:
            best_acc = 100.*tgt_correct/total
            best_model = copy.deepcopy(model)

            
    model = best_model

    return model