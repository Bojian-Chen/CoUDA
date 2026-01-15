import torch
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import scipy.io as sio 
from scipy.fftpack import fft
import torchvision.transforms as transforms
from torch.utils.data import Dataset , DataLoader, TensorDataset, ConcatDataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import argparse

class dataloader():

    def __init__(self, args, session, nb_exemplar, index_exemplar, nb_shot, train = False, few_shot = False, random_exemplar = False):
 
        if train == True:
            dataset=sio.loadmat(args.dataroot+args.train_list) 
        if train == False:
            dataset=sio.loadmat(args.dataroot+args.test_list) 

        self.data = dataset['data'] 
        if args.data_mode == 'Frequence':
            self.data = torch.FloatTensor(abs(fft(self.data)))
        if args.data_mode == 'Time':
            self.data = torch.FloatTensor((self.data))

        #    load label
        self.targets = torch.LongTensor(dataset['label']) 
        self.targets = self.targets.reshape(-1)
     
        self.data, self.targets = self.SelectfromDefault(self.data, self.targets, session, args.Domain_Seq, nb_exemplar, index_exemplar, 
                                                         nb_shot, train, few_shot, random_exemplar)
        

        if args.preprocess == 'zscore':
            mean = torch.mean(self.data, dim=0, keepdim=True)
            std = torch.std(self.data, dim=0, keepdim=True)
            self.data = (self.data - mean) / std
        if args.preprocess == 'minmax':
            min, _ = torch.min(self.data, dim=0, keepdim=True)
            max, _ = torch.max(self.data, dim=0, keepdim=True)
            self.data = (self.data - min) / (max - min)
        if args.preprocess == 'None':
            pass

        if args.data_dimension == '2D':
            self.data = torch.reshape(self.data, [-1, 1, 32, 32])
        if args.data_dimension == '1D':
            self.data = torch.reshape(self.data, [-1, 1, 1024])

    def SelectfromDefault(self, data, targets, session, Domain_Seq, nb_exemplar, index_exemplar, nb_shot, train, few_shot, random_exemplar):
        
        Domain_ID = Domain_Seq[session]
        Replay_Domain_Seq = Domain_Seq[ : session]

        replay_index = []
        few_shot_index = []

        max_label = targets.max()+1

        if train:

            if session == 0:
                ind = np.arange(Domain_ID*max_label*100,(Domain_ID+1)*max_label*100)

            else:


                if few_shot:

                    for k in range(Domain_ID*max_label,(Domain_ID+1)*max_label):

                        b = np.random.choice(range(k*100,(k+1)*100), nb_shot, replace=False)
                        few_shot_index.append(b)

                    few_shot_index = np.hstack(few_shot_index).astype('int64')
                    ind = few_shot_index
                    # print(index)

                else:

                    ind = np.arange(Domain_ID*max_label*100,(Domain_ID+1)*max_label*100)
                    # ind = np.hstack((replay_index,ind)).astype('int64')
        else:
            
            ind = np.arange(Domain_ID*max_label*100,(Domain_ID+1)*max_label*100)
        if nb_exemplar > 0:
            if random_exemplar:
                
                for j in range(session):

                    for i in range(Replay_Domain_Seq[j]*max_label,(Replay_Domain_Seq[j]+1)*max_label):

                        a = np.random.choice(range(i*100,(i+1)*100), nb_exemplar, replace=False)
                        replay_index.append(a)

                ind = np.hstack(replay_index)
                
            else:
                
                if index_exemplar is None:
                    ind = np.random.choice(range(100), 0, replace=False)
                    
                else:
                    ind = np.hstack(index_exemplar)
    
        data_tmp = data[ind]
        targets_tmp = targets[ind]
        
        return data_tmp, targets_tmp




    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self):
        return len(self.data)




if __name__ == "__main__":

    Domain_Seq = np.arange(8)
    nb_session = len(Domain_Seq)

    parser = argparse.ArgumentParser()

    ### Basic parameters
    parser.add_argument('--random_seed', default=2023, type=int, help='random seed')
    parser.add_argument('--backbone_name', default='resnet32', type=str, choices=['resnet32', 'resnet18_1D','cnn'], help='the backbone name')
    parser.add_argument('--batch_size', default=100, type=int, help='the batch size for data loader')
    parser.add_argument('--base_epochs', default=40, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=40, type=int, help='the number of epochs')
    parser.add_argument('--nb_cl', default=5, type=int, help='the number of classes')
    parser.add_argument('--base_lr', default=0.1, type=float, help='the learning rate for base train')
    parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')

    ### Dataset parameters
    parser.add_argument('--data_dimension', default='2D', choices=['1D', '2D'], type=str, help='the dimension of data')
    parser.add_argument('--data_mode', default='Frequence', type=str, choices=['Frequence', 'Time'], help='the mode of data')
    parser.add_argument('--dataroot', default='./data/', type=str, help='the path to load the data')
    parser.add_argument('--train_list', default='./WT_all_5classes.mat', type=str, help='the name of the source dir')
    parser.add_argument('--test_list', default='./WT_all_5classes.mat', type=str, help='the name of the test dir')
    parser.add_argument('--preprocess', default= 'zscore', type=str, choices=['zscore', 'minmax', 'None'], help='the preprocess setting')
    ### Loader parameters
    parser.add_argument('--Domain_Seq', default=Domain_Seq , type=int, help='the Domain_Seq setting')
    parser.add_argument('--nb_exemplar', default=5, type=int, help='the number of exemplars for each class')
    parser.add_argument('--random_exemplar', action='store_false', help='the random exemplar setting')
    parser.add_argument('--index_exemplar', default=None, type=int, help='the random exemplar setting')
    parser.add_argument('--nb_shot', default=0, type=int, help='the number of shots for each class')
    parser.add_argument('--few_shot', action='store_true', help='the few-shot setting')
    parser.add_argument('--nb_session', default=nb_session, type=int, help='the number of sessions')

    args = parser.parse_args()

    # src_trainset = dataloader(args,session=2, nb_exemplar = 0, index_exemplar = None, nb_shot = 0,train= True, few_shot=False, random_exemplar=False)
    # src_trainloader = torch.utils.data.DataLoader(dataset=src_trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # explamars_set = dataloader(args,session=1, nb_exemplar = 5, index_exemplar = [ 97,6 , 66,  55 , 93, 182, 108], nb_shot = 0,train= True, few_shot=False, random_exemplar=False)
    explamars_set= dataloader(args, session=1, nb_exemplar = args.nb_exemplar, index_exemplar = args.index_exemplar, nb_shot = 0,train= True, few_shot=False, random_exemplar=args.random_exemplar)
    # data = explamars_set.data
    # targets = explamars_set.targets
    # print(data.shape)
    # print(targets)
    # combined_clusteroader = DataLoader(ConcatDataset([src_trainloader.dataset ,explamars_set]), batch_size=args.batch_size, shuffle=True, drop_last=False)
    # for btach, input in enumerate(combined_clusteroader) :
    #     print(input[2])




