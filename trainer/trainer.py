import torch
import copy
from utils import *
import os
import models.resnet32 as resnet32
import models.resnet1d as resnet1d
import models.cnn as CNN
import models.modified_linear as modified_linear
from dataloader_domain import dataloader as dataloader
from trainer.base_train import base_train
from trainer.incremental_train import incremental_train


def set_model(args, network):
    model = network(num_classes=args.nb_cl)
    if args.classifer == 'fc':
        pass
    elif args.classifer == 'cos':
        model.fc = modified_linear.CosineLinear(model.fc.in_features * 1, args.nb_cl)
    elif args.classifer == 'eu':
        model.fc = modified_linear.EuclideanLinear(model.fc.in_features * 1, args.nb_cl)
    return model

def set_exemplar(args, session, feature, prediction_label):
    
    feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
    exemplar_index = []
    for i in range(args.nb_cl):
        cl_index = np.where(prediction_label == i)[0]
        if len(cl_index) < args.nb_exemplar:
            continue

        feature_i = feature[cl_index]
        feature_avg = np.mean(feature_i, axis=0)
        D = np.dot(feature_i, feature_avg)
        ind_max = np.argsort(D)[:args.nb_exemplar]
        ind_max = cl_index[ind_max] +  args.Domain_Seq[session]*args.nb_cl*100
        exemplar_index.append(ind_max)

    exemplar_index = np.hstack(np.array(exemplar_index))
    return exemplar_index


def train(args):
    if args.backbone_name == 'resnet32':
        network = resnet32.resnet32
    elif args.backbone_name == 'resnet14':
        network = resnet32.resnet14

    
    model = set_model(args, network)
    Correct = []

    pth = './log/' + args.dataset_name + '/' + args.incremental_mode + '/' + args.preprocess + '/' \
         + str(os.path.splitext(os.path.basename(args.train_list))[0]) + \
        '_' + args.backbone_name + \
        '_' + str(args.contrastive_loss) + \
        '_' + args.classifer + \
        '_' + args.train_parames + \
        '_' + args.data_dimension + \
        '_' + args.data_mode + \
        '_' + str(args.Domain_Seq) + \
        '_' + str(args.random_seed)

    args.pth = pth
    print(args.pth)

    if args.save_model:
        if not os.path.exists(pth):
            os.makedirs(pth)

    for session in range(args.nb_session):
        print('session: {}'.format(session))

        trainset, testset, trainloader, testloader = set_dataset(args, session)
        src_trainset = dataloader(args,session=0, nb_exemplar = 0, index_exemplar = None, nb_shot = 0,train= True, few_shot=False, random_exemplar=False)
        src_trainloader = torch.utils.data.DataLoader(dataset=src_trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        src_testloader = torch.utils.data.DataLoader(dataset=src_trainset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        if args.nb_exemplar > 0:
            exemplar_dataset = dataloader(args, session=session, nb_exemplar = args.nb_exemplar, index_exemplar = args.index_exemplar, nb_shot = 0,train= True, few_shot=False, random_exemplar=args.random_exemplar)

        if args.incremental_mode == 'single':
                model = set_model(args, network)

        tg_optimizer,tg_lr_scheduler = set_optimizer(args,model,session)

        if session == 0:
            model = base_train(args, model, src_trainloader, src_testloader,tg_optimizer,tg_lr_scheduler)
            model_old = copy.deepcopy(model)
            model_src = copy.deepcopy(model)

        else:
            if args.incremental_mode == 'fine_tuning':
                model = base_train(args, model, trainloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'single':
                model = base_train(args, model, trainloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'ours':
                model = incremental_train(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            model_old = copy.deepcopy(model)

        if args.save_model:
            torch.save(model, pth+'/model_session_'+str(session)+'_domain_'+str(args.Domain_Seq[session])+'.pth')


        C =[]
        for k in range(args.nb_session):
            evalset = dataloader(args, session=k, nb_exemplar = 0, index_exemplar = None, nb_shot = 0,train= False, few_shot=False,random_exemplar=False)
            evalloader = torch.utils.data.DataLoader(dataset=evalset, batch_size=args.test_batch_size, shuffle=False, num_workers=0,pin_memory=True)
            correct, features, prediction_label = evaluate(model_old,evalloader,args.Domain_Seq,k)

            if k == session and args.nb_exemplar > 0 and args.random_exemplar == False:
                print('------Herding------')
                exemplar_index = set_exemplar(args, session, features, prediction_label)
                if args.index_exemplar is None:
                    args.index_exemplar = exemplar_index
                else:
                    args.index_exemplar = np.concatenate((args.index_exemplar, exemplar_index))

            if args.save_model:
                torch.save(features,pth+'/features_ours_session_'+str(session)+'_domain_'+str(args.Domain_Seq[k])+'.pth')
            
            C.append(correct)
        Correct.append(C)
        print(Correct)
    return Correct
        
