#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Bojian
"""
import numpy as np
import argparse
import time
import pandas as pd
from utils import *
from trainer.trainer import train

parser = argparse.ArgumentParser()

### Basic parameters
parser.add_argument('--random_seed', default=2024, type=int, help='random seed')
parser.add_argument('--backbone_name', default='resnet14', type=str, choices=['resnet14','resnet32'], help='the backbone name')
parser.add_argument('--nb_cl', type=int, help='the number of classes')
parser.add_argument('--batch_size', default=128, type=int, help='the batch size for data loader')
parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test data loader')

### Train parameters
parser.add_argument('--base_epochs', default=40, type=int, help='the number of epochs in base train')
parser.add_argument('--epochs', default=40, type=int, help='the number of epochs')
parser.add_argument('--base_lr', default=0.1, type=float, help='the learning rate for base train')
parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')
parser.add_argument('--contrastive_loss', action='store_false', help='the contrastive loss setting')
parser.add_argument('--temperature', default=0.07, type=float, help='the temperature for contrastive loss')
parser.add_argument('--weight_contrastive', default=0.1, type=float, help='the weight for contrastive loss')
parser.add_argument('--preprocess', default= 'zscore', type=str, choices=['zscore', 'minmax', 'None'], help='the preprocess setting')

### Incremental parameters
parser.add_argument('--incremental_mode', default='ours', type=str, choices=['fine_tuning', 'single', 'ours', 'MMD','LMMD','MMDA','DANN','IDANN','CORAL', 'CUA', 'CUA_MMD','ConDA','DCTLN_DWA','MuHDi'], help='the incremental mode')
parser.add_argument('--classifer', default='cos', type=str, choices=['fc', 'cos', 'eu'], help='the classifier')
parser.add_argument('--train_parames', default='extractor', type=str, choices=['all', 'extractor', 'classifier', 'layer1_2', 'layer2_fc', 'BN', 'woBN'], help='the trained parameters of model')

### Dataset parameters
parser.add_argument('--dataset_name', default='SK', type=str, choices=['SK'], help='the dataset name')
parser.add_argument('--data_dimension', default='2D', choices=['1D', '2D'], type=str, help='the dimension of data')
parser.add_argument('--data_mode', default='Frequence', type=str, choices=['Frequence', 'Time'], help='the mode of data')
parser.add_argument('--dataroot', default='./data/', type=str, help='the path to load the data')
parser.add_argument('--train_list',type=str, help='the name of the source dir')
parser.add_argument('--test_list',  type=str, help='the name of the test dir')

### Dataloader parameters
parser.add_argument('--Domain_Seq',  type=int, help='the Domain_Seq setting')
parser.add_argument('--nb_session', type=int, help='the number of sessions')
parser.add_argument('--nb_exemplar', default=0, type=int, help='the number of exemplars for each class')
parser.add_argument('--random_exemplar', action='store_true', help='the random exemplar setting')
parser.add_argument('--index_exemplar', default=None, type=int, help='the index of exemplar')
parser.add_argument('--nb_shot', default=0, type=int, help='the number of shots for each class')
parser.add_argument('--few_shot', action='store_true', help='the few-shot setting')


### Save and Draw
parser.add_argument('--save_model', action='store_true', help='the save setting')
parser.add_argument('--draw', action='store_true', help='the draw setting')

args = parser.parse_args()

### Set Domain Setting
if args.dataset_name == 'SK':
    args.train_list = './SK_all_10classes.mat'
    args.test_list = './SK_all_10classes.mat'
    args.Domain_Seq = np.array([6,1,8,15,22,17])  # 转速 负载 持续变化
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 10

print('=============================================================================================================')
print(args)
print('=============================================================================================================')
print( args.dataset_name)
print('=============================================================================================================')
### Set the random seed
random_seed(args.random_seed)

time_start=time.time()
Correct = train(args)
time_end=time.time()

BWT = []
AA = []
AF = []
AG = []
Correct_t = torch.tensor(Correct)

for i in range(Correct_t.shape[0]):
    if i > 0:
        AA.append(Correct_t[i,i])
        BWT.append( Correct[len(Correct)-1][i] - Correct[i][i])

for j in range(Correct_t.shape[0]-2):

    AF.append( Correct_t[j+2:Correct_t.shape[0],j+1])
    AG.append( Correct_t[j,j+1:Correct_t.shape[0]])

AF = torch.mean(torch.cat(AF)).item()
AMF = (AF - torch.mean(torch.tensor(AA[:-1]))).item()
AG = torch.mean(torch.cat(AG)).item()
AA = torch.mean(torch.tensor(AA)).item()
BWT = torch.mean(torch.tensor(BWT)).item()
ACC = torch.mean(Correct_t[len(Correct) - 1, 1:]).item()

print('AF: {:.2f}%'.format(AF))
print('AMF: {:.2f}%'.format(AMF))
print('AG: {:.2f}%'.format(AG))
print('AA: {:.2f}%'.format(AA))
print('BWT: {:.2f}%'.format(BWT))
print('ACC: {:.2f}%'.format(ACC))

if args.save_model:
    results = [
        {'Metric': 'AF', 'Value': AF},
        {'Metric': 'AMF', 'Value': AMF},
        {'Metric': 'AG', 'Value': AG},
        {'Metric': 'AA', 'Value': AA},
        {'Metric': 'BWT', 'Value': BWT},
        {'Metric': 'ACC', 'Value': ACC}
    ]
    df = pd.DataFrame(results)
    df1 = pd.DataFrame(Correct)
    df = pd.concat([df, df1], axis=1)
    df.to_csv(args.pth + '/Result.csv', index=False)

print('totally cost: {:.2f}s'.format(time_end-time_start))






