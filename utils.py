import torch
import numpy as np
import torch.nn as nn
import random
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.optim as optim
from dataloader_domain import dataloader as dataloader
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix 
import os
import matplotlib.colors as mcolors
def set_dataset(args, session):
    if session ==0:

        trainset = dataloader(args, session=session, nb_exemplar = 0, index_exemplar = None, nb_shot = 0, train= True, few_shot=False, random_exemplar=False)
        testset = dataloader(args, session=session, nb_exemplar = 0, index_exemplar = None, nb_shot = 0,train= False, few_shot=False,random_exemplar=False)
        
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0,pin_memory=True)

    else:

        trainset = dataloader(args, session=session, nb_exemplar = 0, index_exemplar =  None, nb_shot =  args.nb_shot,train= True, few_shot= args.few_shot,random_exemplar= False)
        testset = dataloader(args, session=session, nb_exemplar = 0, index_exemplar = None, nb_shot = 0,train= False, few_shot=False,random_exemplar=False)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0,pin_memory=True)

    return trainset, testset, trainloader, testloader




def set_optimizer(args,model,session):

    if session == 0:
        
        tg_optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
        lr_strat = [int(args.base_epochs*0.5), int(args.base_epochs*0.75)]
        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=0.1)

    else:
        if args.train_parames == 'all':
            tg_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.train_parames == 'extractor':
            ignored_params = list(map(id, model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            base_params = filter(lambda p: p.requires_grad,base_params)
            target_params = [{'params': base_params, 'lr': args.lr, 'weight_decay': 5e-4}]
            tg_optimizer = optim.SGD(target_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.train_parames == 'classifier':
            target_params = [{'params': model.fc.parameters(), 'lr': args.lr, 'weight_decay': 5e-4}]
            tg_optimizer = optim.SGD(target_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.train_parames == 'layer1_2':
            for param in model.layer3.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = False
            base_params = filter(lambda p: p.requires_grad, model.parameters())
            target_params = [{'params': base_params, 'lr': args.lr, 'weight_decay': 5e-4}]
            tg_optimizer = optim.SGD(target_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.train_parames == 'layer2_fc':
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
            # for param in model.layer2.parameters():
            #     param.requires_grad = False
            base_params = filter(lambda p: p.requires_grad, model.parameters())
            target_params = [{'params': base_params, 'lr': args.lr, 'weight_decay': 5e-4}]
            tg_optimizer = optim.SGD(target_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

        elif args.train_parames == 'BN':
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    param.requires_grad = False
            base_params = filter(lambda p: p.requires_grad, model.parameters())
            target_params = [{'params': base_params, 'lr': args.lr, 'weight_decay': 5e-4}]
            tg_optimizer = optim.SGD(target_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

        elif args.train_parames == 'woBN':
            for name, param in model.named_parameters():
                if 'bn' in name:
                    param.requires_grad = False
            base_params = filter(lambda p: p.requires_grad, model.parameters())
            target_params = [{'params': base_params, 'lr': args.lr, 'weight_decay': 5e-4}]
            tg_optimizer = optim.SGD(target_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

        lr_strat = [int(args.epochs*0.5), int(args.epochs*0.75)]
        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=0.1)

      
    return tg_optimizer,tg_lr_scheduler


def evaluate(model_old, evalloader, Domain_Seq, k, weight_per_class=None, device = None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_samples = evalloader.dataset.data.shape[0]
    num_features = 64

    features_storage = np.zeros([num_samples, num_features])

    model = model_old.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    prediction_label = []
    Targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs, features = model(inputs)
            features_storage[ batch_idx*100:batch_idx*100+inputs.shape[0], :] = features.cpu().numpy()



            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            prediction_label.append(predicted.cpu())
            Targets.append(targets.cpu())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    prediction_label = torch.Tensor( np.array( [item.numpy() for item in prediction_label])).reshape(-1)
    Targets = torch.Tensor( np.array( [item.numpy() for item in Targets])).reshape(-1)

    cm = confusion_matrix(prediction_label,Targets)
    print(cm)
    print('eval domain: {} eval set: {} test loss: {:.4f} accuracy: {:.4f}'.format(Domain_Seq[k], len(evalloader), test_loss/(batch_idx+1), 100.*correct/total))
    return 100.*correct/total, features_storage , prediction_label


def random_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def get_pseudo_data_and_label(model_old,testloader,embedding):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_old = model_old.to(device)
    model_old.eval()
    tg_feature_model = nn.Sequential(*list(model_old.children())[:-1])
    
    correct = 0
    correct_1 = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_old(inputs)
            features = tg_feature_model(inputs).reshape(-1,64)
            softmax_outputs = F.softmax(outputs.detach(), dim=1)
            soft_output,_ = softmax_outputs.max(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            
            icarl_cosine = cdist(embedding, features.cpu(), 'cosine')
            score_icarl_cosine = torch.from_numpy((-icarl_cosine).T).to(device)

            icarl, predicted_icarl_cosine = score_icarl_cosine.max(1)
            correct_1 += predicted_icarl_cosine.eq(targets).sum().item()
            
            print(targets)
            print(predicted)
            print(predicted.eq(targets))
            print(soft_output)

    return predicted



def compute_features(model,evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            outputs, the_feature = model(inputs)
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(the_feature.cpu())
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features

def compute_current_session_embedding(args, model, trainset):
    # tg_feature_model = nn.Sequential(*list(model.children())[:-1])
    # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
    num_features = model.fc.in_features
    num_samples = trainset.data.shape[0]
    tem_trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=False, num_workers=0,pin_memory=True)
    cls_features = compute_features(model,tem_trainloader, num_samples, num_features).reshape(args.nb_cl,-1,num_features)
    norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=2)
    cls_prototype = torch.mean(cls_features, dim=1)
    # novel_embedding = F.normalize(cls_embedding, p=2, dim=1)
    return norm_features, cls_prototype
            
def compute_embedding_and_set_exemplars(args, model, trainset, testset,session,nb_exemplar):
    model.eval()

    tg_feature_model = nn.Sequential(*list(model.children())[:-1])
    # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
    num_features = model.fc.in_features
    # Intialize the new FC weights with zeros
    embedding = torch.zeros((args.nb_cl, num_features))
    
    fc_para = [para for para in model.fc.named_parameters()]

    embedding_para = fc_para[0][1].data.cpu()
    # print(embedding_para.shape)
    embedding_para = F.normalize(embedding_para, p=2, dim=1)
    
    # data_feature = compute_features( tg_feature_model,tem_evalloader, num_samples, num_features)
    # a = cdist(data_feature,embedding_para, 'cosine')



    index = []
    for cls_idx in range(args.nb_cl):

        # Set a temporary dataloader for the current class

        testset.data = trainset.data[cls_idx*100:(cls_idx+1)*100]
        testset.targets = trainset.targets[cls_idx*100:(cls_idx+1)*100]
        tem_evalloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False, num_workers=0,pin_memory=True)

        num_samples = testset.data.shape[0]
        # Compute the feature maps using the current model
        cls_features = compute_features( tg_feature_model,tem_evalloader, num_samples, num_features)

        # Compute the normalized feature maps 
        norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
        cls_embedding = torch.mean(torch.from_numpy(cls_features), dim=0) 
  
        D = np.dot(norm_features,embedding_para[cls_idx])
        # print(D[0])
        ind_max = np.argsort(D)[:nb_exemplar]
        # print(ind_max)
        ind_max = [inx+cls_idx*100+args.Domain_Seq[session+1]*1000 for inx in ind_max]
        index.append(ind_max)
        index_exemplar = np.hstack(np.array(index))

        embedding[cls_idx] = cls_embedding

    return embedding,index_exemplar



def compute_best_5_dataloader(model, trainloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    best_5_samples = [[] for _ in range(10)]  # List to store the best 5 samples for each class
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            scores = F.softmax(outputs.detach(), dim=1)
            for i in range(len(targets)):
                class_idx = targets[i].item()
                sample_score = scores[i][class_idx].item()
                best_5_samples[class_idx].append((inputs[i], targets[i], sample_score))
    
    best_5_data = []
    best_5_targets = []
    for class_samples in best_5_samples:
        class_samples.sort(key=lambda x: x[2], reverse=True)  # Sort samples by score in descending order
        best_5_data.extend([sample[0] for sample in class_samples[:5]])
        best_5_targets.extend([sample[1] for sample in class_samples[:5]])
    
    best_5_dataset = torch.utils.data.TensorDataset(torch.stack(best_5_data), torch.tensor(best_5_targets))
    best_5_dataloader = torch.utils.data.DataLoader(dataset=best_5_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)
    
    return best_5_dataloader
