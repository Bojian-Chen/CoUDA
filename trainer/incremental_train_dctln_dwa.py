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
from torch.autograd import Function
from scipy.fftpack import fft
from scipy.spatial.distance import cdist
import torch.utils.data as Data
import matplotlib.pyplot as plt
from loss_function import *
import copy

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

    
class Domain(nn.Module):
    def __init__(self, domain_classes = 2):
        super(Domain, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64*1, 2))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        # self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, domain_classes))

    def forward(self, feature):
        feature = ReverseLayerF.apply(feature, -1)
        class_output = self.class_classifier(feature)
        return class_output

def incremental_train_dctln_dwa(args, model, model_old, src_trainloader, tgt_trainloader, src_testloader , testloader, exemplar_dataset, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):

    best_acc = 0
    best_model = None
    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    examplar_data = exemplar_dataset.data
    examplar_label = exemplar_dataset.targets
    examplar_data = examplar_data.to(device)
    examplar_label = examplar_label.to(device)



        
    model = model.to(device)
    model_old = model_old.to(device)

    discriminator = Domain().to(device)
    dis_optimizer = optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # lr_strat = [int(args.epochs*0.2), int(args.epochs*0.5), int(args.epochs*0.75)]
    lr_strat = [int(args.epochs*0.75)]
    dis_lr_scheduler = lr_scheduler.MultiStepLR(dis_optimizer, milestones=lr_strat, gamma=0.1)


    for epoch in range(args.epochs):
        # Set the model to the training mode
        model.train()
        model_old.eval()
        discriminator.train()
        # model_new.train()


        # Set all the losses to zeros
        train_loss = 0
        train_loss_cls = 0
        train_loss_domain = 0
        train_loss_mmd = 0
        train_loss_distillation = 0

        # Set the counters to zeros
        src_correct = 0
        tgt_correct = 0

        total = 0

        # Print the information
        for batch_idx,((source_data, source_label),(target_data, target_label)) in enumerate(zip(src_trainloader,tgt_trainloader)): 
            batch_size = len(source_data)

            s_domain_label = torch.zeros(batch_size).long()
            t_domain_label = torch.ones(batch_size).long()

            source_data, source_label, target_data, target_label = source_data.to(device), source_label.to(device), target_data.to(device), target_label.to(device)
            s_domain_label, t_domain_label = s_domain_label.to(device), t_domain_label.to(device)

            tg_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            src_outputs, src_features = model(source_data)
            src_domain_output = discriminator(src_features)
            tgt_outputs, tgt_features = model(target_data)
            tgt_domain_output = discriminator(tgt_features)

            # inputs = torch.cat((source_data, target_data), 0)


            # # Forward the samples in the deep networks
            # new_outputs, new_features = model(inputs)
            # domain_output = discriminator(new_features.detach())

            # # print(new_features.shape)
            # _, src_predicted = new_outputs[ : batch_size].max(1)
            # _, tgt_predicted = new_outputs[batch_size : ].max(1)
            # _, src_d_predicted = domain_output[ : batch_size].max(1)
            # _, tgt_d_predicted = domain_output[batch_size : ].max(1)
            _, src_predicted = src_outputs.max(1)
            _, tgt_predicted = tgt_outputs.max(1)
            _, src_d_predicted = src_domain_output.max(1)
            _, tgt_d_predicted = tgt_domain_output.max(1)


            #Compute classification loss
            # loss1 = nn.CrossEntropyLoss(weight_per_class)(new_outputs[ : batch_size], source_label) 
            # loss2 = nn.CrossEntropyLoss(weight_per_class)(domain_output[ : batch_size], s_domain_label)
            # loss3 = nn.CrossEntropyLoss(weight_per_class)(domain_output[batch_size : ], t_domain_label)
            loss1 = nn.CrossEntropyLoss(weight_per_class)(src_outputs, source_label) 
            loss2 = nn.CrossEntropyLoss(weight_per_class)(src_domain_output, s_domain_label)
            loss3 = nn.CrossEntropyLoss(weight_per_class)(tgt_domain_output, t_domain_label)

            T = 2
            exemplar_outputs, _ = model(examplar_data)
            old_exemplar_outputs, _ = model_old(examplar_data)
            loss4 = nn.KLDivLoss()(F.log_softmax(exemplar_outputs/T, dim=1), F.softmax(old_exemplar_outputs/T, dim=1)) * T * T
                                   
            loss = loss1 + 0.5*(loss2+ loss3) + loss4  + mmd(src_features,tgt_features)*0.1
            
            loss.backward()
            tg_optimizer.step()
            dis_optimizer.step()




            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss_cls += loss1.item()
            train_loss_domain+= 0.5*(loss2.item() + loss3.item())
            train_loss_distillation += loss4.item()
            total += source_label.size(0)
            src_correct += src_predicted.eq(source_label).sum().item()
            tgt_correct += tgt_predicted.eq(target_label).sum().item()

   
        # Learning rate decay
        tg_lr_scheduler.step()

        
        # Print the training losses and accuracies

        print('Train set: {} train loss: {:.4f}  cls {:.4f}  domain {:.4f}   distillation {:.4f} src_accuracy: {:.4f} tgt_accuracy: {:.4f} '.format(
            len(src_trainloader), 
            train_loss/(batch_idx+1), 
            train_loss_cls/(batch_idx+1), 
            train_loss_domain/(batch_idx+1),
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