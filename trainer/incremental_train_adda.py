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
import copy


class Domain(nn.Module):
    def __init__(self, domain_classes = 2):
        super(Domain, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64*1, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, domain_classes))

    def forward(self, feature):
        class_output = self.class_classifier(feature)
        return class_output
    
class Mymodel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(Mymodel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        x = self.feature_extractor(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(feature)
        return x, feature
    
def incremental_train_adda(args, model, model_src, src_trainloader, tgt_trainloader, src_testloader , testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    source_feature_model = nn.Sequential(*list(model_src.children())[:-1])
    target_feature_model = nn.Sequential(*list(model.children())[:-1])
    discriminator = Domain()

    source_feature_model = source_feature_model.to(device)
    target_feature_model = target_feature_model.to(device)
    discriminator = discriminator.to(device)

    tgt_optimizer = optim.SGD(target_feature_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    dis_optimizer = optim.SGD(discriminator.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # lr_strat = [int(args.epochs*0.5), int(args.epochs*0.75)]
    # dis_lr_scheduler = lr_scheduler.MultiStepLR(dis_optimizer, milestones=lr_strat, gamma=0.1)


    for epoch in range(args.epochs):
        # Set the model to the training and evaluation mode
        source_feature_model.eval()
        target_feature_model.train()
        discriminator.train()

        # Set all the losses to zeros
        train_loss_dis = 0
        train_loss_fea = 0

        # Set the counters to zeros
        correct_1 = 0
        correct_2 = 0
        total = 0

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        for batch_idx,((source_data, source_label),(target_data, target_label)) in enumerate(zip(src_trainloader,tgt_trainloader)): 
            batch_size = len(source_data)
            # Train discriminator
            s_domain_label = torch.zeros(batch_size).long()  # source 0
            t_domain_label = torch.ones(batch_size).long()   # traget 1
            label_concat = torch.cat((s_domain_label, t_domain_label), 0)

            source_data, source_label, target_data, target_label = source_data.to(device), source_label.to(device), target_data.to(device), target_label.to(device)
            label_concat = label_concat.to(device)

            dis_optimizer.zero_grad()
            source_feature = source_feature_model(source_data)
            source_feature = source_feature.view(source_feature.size(0), -1)
            target_feature = target_feature_model(target_data)
            target_feature = target_feature.view(target_feature.size(0), -1)

            features_concat = torch.cat((source_feature, target_feature), 0)


            domain_output = discriminator(features_concat)
            loss1 = nn.CrossEntropyLoss(weight_per_class)(domain_output, label_concat)
            loss1.backward()
            dis_optimizer.step()

            domain_predicted = domain_output.max(1)[1]
            correct_1 += domain_predicted.eq(label_concat).sum().item()
            
            # Train target feature extractor
            tgt_optimizer.zero_grad()
            # dis_optimizer.zero_grad()

            target_feature = target_feature_model(target_data)
            target_feature = target_feature.view(target_feature.size(0), -1)
            target_domain_output = discriminator(target_feature)

            t_domain_label_fake = torch.zeros(batch_size).long()  # target 0
            t_domain_label_fake = t_domain_label_fake.to(device)

            loss2 = nn.CrossEntropyLoss(weight_per_class)(target_domain_output, t_domain_label_fake)
            loss2.backward()
            tgt_optimizer.step()

            target_domain_predicted = target_domain_output.max(1)[1]
            correct_2 += target_domain_predicted.eq(t_domain_label_fake).sum().item()
            
            # Record the losses and the number of samples to compute the accuracy
            train_loss_dis += loss1.item()
            train_loss_fea += loss2.item()
            total += source_label.size(0)

        # Learning rate decay
        print(args.lr)

        # Print the training losses and accuracies
        print('Train set: {}, train loss dis: {:.4f} train loss fea: {:.4f} dis_accuracy: {:.4f} tgt_accuracy: {:.4f}'.format(
            len(src_trainloader), train_loss_dis/(batch_idx+1), train_loss_fea/(batch_idx+1), 100.*correct_1/(total*2), 100.*correct_2/total))
        
        # Running the test for this epoch
       

        source_classifier_model = list(model_src.children())[-1]

        model_cont = Mymodel(target_feature_model, source_classifier_model)
        model_cont = model_cont.to(device)
        model_cont.eval()

        # model = nn.Sequential(target_feature_model, source_classifier_model)

        test_loss = 0
        correct = 0

        total = 0
        with torch.no_grad():
            for batch_idx,(data, label) in enumerate(testloader): 

                data, label = data.to(device), label.to(device)

                outputs,_ = model_cont(data)
                _, predicted = outputs.max(1)
                correct += predicted.eq(label).sum().item()
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, label)

                test_loss += loss.item()
                total += label.size(0)
                

        print('Test set: {} test loss: {:.4f} accuracy: {:.4f} '.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
    model = copy.deepcopy(model_cont)

    return model