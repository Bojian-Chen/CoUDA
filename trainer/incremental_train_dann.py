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
from torch.optim import lr_scheduler
from torch.autograd import Function
from loss_function import *

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

def incremental_train_dann(args, model, src_trainloader, tgt_trainloader, src_testloader , testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    discriminator = Domain().to(device)
    dis_optimizer = optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # lr_strat = [int(args.epochs*0.2), int(args.epochs*0.5), int(args.epochs*0.75)]
    lr_strat = [int(args.epochs*0.75)]
    dis_lr_scheduler = lr_scheduler.MultiStepLR(dis_optimizer, milestones=lr_strat, gamma=0.1)


    for epoch in range(args.epochs):
        # Set the model to the training mode
        model.train()
        discriminator.train()
        

        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        # Set the counters to zeros
        correct = 0
        correct_1 = 0
        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
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
            loss = loss1 + 0.5*(loss2+ loss3)
            
            loss.backward()
            tg_optimizer.step()
            dis_optimizer.step()




            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
 
            total += source_label.size(0)
            correct += src_predicted.eq(source_label).sum().item()
            correct_1 += tgt_predicted.eq(target_label).sum().item()

        # Learning rate decay
        tg_lr_scheduler.step()
        # dis_lr_scheduler.step()
        # Print the training losses and accuracies
        print(tg_lr_scheduler.get_last_lr()[0])
        print('Train set: {}, train loss: {:.4f} src_accuracy: {:.4f} tgt_accuracy: {:.4f}'.format(len(src_trainloader), train_loss/(batch_idx+1), 100.*correct/total, 100.*correct_1/total))
        
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