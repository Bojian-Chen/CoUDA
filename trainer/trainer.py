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
from trainer.incremental_train_idann import incremental_train_idann
from trainer.incremental_train_dann import incremental_train_dann
from trainer.incremental_train_adda import incremental_train_adda
from trainer.incremental_train_spilt import incremental_train_spilt
from trainer.incremental_train_source_free import incremental_train_source_free
from trainer.incremental_train_mmd import incremental_train_mmd
from trainer.incremental_train_lmmd import incremental_train_lmmd
from trainer.incremental_train_coral import incremental_train_coral
from trainer.incremental_train_cua import incremental_train_cua
from trainer.incremental_train_dctln_dwa import incremental_train_dctln_dwa
from trainer.incremental_train_cua_mmd import incremental_train_cua_mmd
from trainer.incremental_train_mmda import incremental_train_mmda
from trainer.incremental_train_everadapt import incremental_train_everadapt
from trainer.incremental_train_dctln_MuHDi import incremental_train_MuHDi
from trainer.incremental_train_ConDA import incremental_train_ConDA

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
    # feature = feature.cpu().numpy()
    
    feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
    exemplar_index = []
    for i in range(args.nb_cl):
        cl_index = np.where(prediction_label == i)[0]
        if len(cl_index) < args.nb_exemplar:
            continue
        # print(cl_index)
        feature_i = feature[cl_index]
        # feature_i = feature[i*100:(i+1)*100]
        feature_avg = np.mean(feature_i, axis=0)
        D = np.dot(feature_i, feature_avg)
        ind_max = np.argsort(D)[:args.nb_exemplar]
        ind_max = cl_index[ind_max] +  args.Domain_Seq[session]*args.nb_cl*100
        # print(ind_max)
        # ind_max = [inx+i*100+args.Domain_Seq[session]*args.nb_cl*100 for inx in ind_max]
        exemplar_index.append(ind_max)
    exemplar_index = np.hstack(np.array(exemplar_index))
    return exemplar_index


def train(args):
    if args.backbone_name == 'resnet32':
        network = resnet32.resnet32
    elif args.backbone_name == 'resnet14':
        network = resnet32.resnet14
    elif args.backbone_name == 'resnet18_1D':
        network = resnet1d.resnet18
    elif args.backbone_name == 'cnn':
        network = CNN.CNN

    
    model = set_model(args, network)

    # weights = model.fc.weight.data
    # print(weights)
    Correct = []
    # print(os.path.splitext(os.path.basename(args.train_list))[0] )
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
    # print(pth)
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
            # exemplar_loader = torch.utils.data.DataLoader(dataset=exemplar_dataset, batch_size=, shuffle=False, num_workers=0, pin_memory=True)

        if args.incremental_mode == 'single':
                model = set_model(args, network)

        tg_optimizer,tg_lr_scheduler = set_optimizer(args,model,session)

        if session == 0:
            model = base_train(args, model, src_trainloader, src_testloader,tg_optimizer,tg_lr_scheduler)
            model_old = copy.deepcopy(model)
            model_src = copy.deepcopy(model)
            # features, prototypes = compute_current_session_embedding(args, model_old, trainset)
        else:
            if args.incremental_mode == 'fine_tuning':
                model = base_train(args, model, trainloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'single':
                model = base_train(args, model, trainloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'ours':
                model = incremental_train(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'MMD':
                model = incremental_train_mmd(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'LMMD':
                model = incremental_train_lmmd(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'MMDA':
                model = incremental_train_mmda(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'CORAL':
                model = incremental_train_coral(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'DANN':
                model = incremental_train_dann(args, model, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'IDANN':
                model = incremental_train_idann(args, model, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)

            elif args.incremental_mode == 'CUA':
                model = incremental_train_cua(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader, exemplar_dataset, tg_optimizer, tg_lr_scheduler)

            elif args.incremental_mode == 'CUA_MMD':
                model = incremental_train_cua_mmd(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader, exemplar_dataset, tg_optimizer, tg_lr_scheduler)

            elif args.incremental_mode == 'DCTLN_DWA':
                model = incremental_train_dctln_dwa(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader, exemplar_dataset, tg_optimizer, tg_lr_scheduler)

            elif args.incremental_mode == 'MuHDi':
                model = incremental_train_MuHDi(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader, tg_optimizer, tg_lr_scheduler)

            elif args.incremental_mode == 'EverAdapt':
                model = incremental_train_everadapt(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader, exemplar_dataset, tg_optimizer, tg_lr_scheduler)

            elif args.incremental_mode == 'ADDA':
                model = incremental_train_adda(args, model, model_src, src_trainloader, trainloader, src_testloader , testloader, tg_optimizer, tg_lr_scheduler, weight_per_class=None, device = None)

            elif args.incremental_mode == 'spilt':
                model = incremental_train_spilt(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,exemplar_dataset, tg_optimizer, tg_lr_scheduler)
                
            elif args.incremental_mode == 'source_free':
                model = incremental_train_source_free(args, model, model_old, trainloader , testloader, tg_optimizer, tg_lr_scheduler)
            elif args.incremental_mode == 'ConDA':
                model = incremental_train_ConDA(args, model, model_old, src_trainloader, trainloader , src_testloader, testloader, exemplar_dataset,tg_optimizer, tg_lr_scheduler)
            # model, src_correct, tgt_correct = incremental_train(args, model, model_old, src_trainloader, trainloader, src_testloader, testloader,tg_optimizer,tg_lr_scheduler)
            model_old = copy.deepcopy(model)
            # model_new = copy.deepcopy(model)
        if args.save_model:
            torch.save(model, pth+'/model_session_'+str(session)+'_domain_'+str(args.Domain_Seq[session])+'.pth')
        # torch.save(norm_features,pth+'/ours_session_'+str(session)+'_domain_'+str(args.Domain_Seq[session])+'.pth')
        # torch.save(cls_embedding,pth+'/ours_session_'+str(session)+'_domain_'+str(args.Domain_Seq[session])+'.pth')

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
                # print(args.index_exemplar)

            
            # 特征聚类
            # features, prototypes = compute_current_session_embedding(args, model_old, evalset)
            # features = compute_features(model,evalloader, nb_cl*100, 64)
            if args.save_model:
                torch.save(features,pth+'/features_ours_session_'+str(session)+'_domain_'+str(args.Domain_Seq[k])+'.pth')
            
            C.append(correct)
        Correct.append(C)
        print(Correct)
    return Correct
        
