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
    # print(prediction_label)
    # print(Targets)
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
            # print('inputs', inputs.shape)
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
            # print(icarl_cosine.shape)
            icarl, predicted_icarl_cosine = score_icarl_cosine.max(1)
            correct_1 += predicted_icarl_cosine.eq(targets).sum().item()
            
            print(targets)
            print(predicted)
            print(predicted.eq(targets))
            print(soft_output)
            # print(soft_output.gt(0.95).sum())

            # print(predicted_icarl_cosine)
            # print(predicted_icarl_cosine.eq(targets))
            # print(predicted_icarl_cosine.gt(0.95).sum())

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
        # print(tem_evalloader.dataset.data.shape)
        num_samples = testset.data.shape[0]
        # Compute the feature maps using the current model
        cls_features = compute_features( tg_feature_model,tem_evalloader, num_samples, num_features)

        # Compute the normalized feature maps 
        norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)#dim = 1 按列操作 每行都除以该行下所有元素平方和的开方
        cls_embedding = torch.mean(torch.from_numpy(cls_features), dim=0) #dim = 0 按行操作 求每列的根平均值
        # print(cls_embedding)
        # print(F.normalize(cls_embedding, p=2, dim=0))
        # print(embedding_para[cls_idx])
        
        # a = torch.from_numpy((-a).T)
        # _, b = a.max(1)
        # print(a[0])
            # icarl_cosine = cdist(embedding, features.cpu(), 'cosine')
            # score_icarl_cosine = torch.from_numpy((-icarl_cosine).T).to(device)
            # print(icarl_cosine.shape)
            # _, predicted_icarl_cosine = score_icarl_cosine.max(1)
            # correct_1 += predicted_icarl_cosine.eq(targets).sum().item()
        D = np.dot(norm_features,embedding_para[cls_idx])
        # print(D[0])
        ind_max = np.argsort(D)[:nb_exemplar]
        # print(ind_max)
        ind_max = [inx+cls_idx*100+args.Domain_Seq[session+1]*1000 for inx in ind_max]
        index.append(ind_max)
        index_exemplar = np.hstack(np.array(index))
        # print(index_exemplar)

        # embedding[cls_idx] = F.normalize(cls_embedding, p=2, dim=0) #dim = 0 按行操作 每列都除以该列下所有元素平方和的开方
        embedding[cls_idx] = cls_embedding

    return embedding,index_exemplar



def draw(args,Correct):
    labels = args.Domain_Seq.tolist()
    labely = args.Domain_Seq.tolist()

    tick_marks = np.array(range(len(labels)))+0.5
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.viridis):  #viridis
     
        # print(cm)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title, fontsize=12,family='Times New Roman')
        # plt.colorbar()
        xlocations = np.array(range(len(labels)))
        ylocations = np.array(range(len(labely)))
        plt.xticks(xlocations, labels, rotation=0,fontsize=8,family='Times New Roman',ha='center')
        plt.yticks(ylocations, labely,fontsize=8,family='Times New Roman',va='center')
        plt.ylabel('Domain (train)',fontsize=10,family='Times New Roman',va='bottom')
        plt.xlabel('Domain (test)',fontsize=10,family='Times New Roman',va='top')
    # def plt_confusion_matrix_3D(cm,x,y,dx,dy):
    #     fig = plt.figure()
    #     # ax = fig.add_subplot(projection='3d')
    #     ax = fig.add_subplot(111,projection='3d')
    #     dz = cm.flatten()
    #     ax.bar3d(x,y,cm,dx,dy,dz)
    
    cm_normalized = np.array(Correct)
    plt.figure(dpi=600, figsize=(3,3),frameon=False,facecolor='white',edgecolor='white') 
    ind_array = np.arange(len(labels))
    ind_array_y = np.arange(len(labely))
    x, y = np.meshgrid(ind_array, ind_array_y)
    # x = np.array([[0],[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,5]])
    # print(x.flatten(),y.flatten())
    # plt_confusion_matrix_3D(cm_normalized,x,y,ind_array,ind_array)

    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c <= 50.0:
            # plt.text(x_val, y_val, "%0.1f%s" % (c,*'%') , color='white', fontsize=10, va='center', ha='center',family='Times New Roman')
            plt.text(x_val, y_val, c , color='white', fontsize=6, va='center', ha='center',family='Times New Roman')
        if c > 50.0:
            plt.text(x_val, y_val, c , color='black', fontsize=6, va='center', ha='center',family='Times New Roman')
    # offset the tick
    line_width = 1
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().tick_params(which='minor', left = False, bottom=False)

    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['top'].set_linewidth(line_width)
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['right'].set_linewidth(line_width)
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['bottom'].set_linewidth(line_width)
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['left'].set_linewidth(line_width)
    plt.grid(True,c='white', which='minor', linestyle='-',linewidth =str(line_width))

    # plot_confusion_matrix(cm_normalized, title='Fewshot='+str(nb_shot)+', KD + examplar = '+str(nb_exemplar))
    # plot_confusion_matrix(cm_normalized, title='Knowledge distill+ exemplar = '+str(nb_exemplar))
    plot_confusion_matrix(cm_normalized)
    # name = args.plt_name
    name =  args.incremental_mode + \
        '_' + str(os.path.splitext(os.path.basename(args.train_list))[0]) + \
        '_' + args.backbone_name + \
        '_' + str(args.contrastive_loss) + \
        '_' + args.classifer + \
        '_' + args.train_parames + \
        '_' + args.data_dimension + \
        '_' + args.data_mode + \
        '_' + str(args.Domain_Seq) + \
        '_' + str(args.random_seed )
    # plt.tight_layout()
    # plt.savefig('./fig/new/'+name+'.svg',format='svg')
    plt.show()
    # 


# def positive_sample(model, trainloader, src_testloader, epoch):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
#     positive_sample_index = []
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs, _ = model(inputs)
#             cos_sim, predicted = outputs.max(1)
#             # softmax_outputs,_ = F.softmax(outputs.detach(), dim=1).max(1)
#             ind_max = np.argsort(cos_sim.cpu())[:epoch] + batch_idx * 100

#             positive_sample_index.append(ind_max.numpy())

#     positive_sample_index = np.hstack(positive_sample_index)
#     positive_sample_data, positive_sample_targets = testloader.dataset.data[positive_sample_index], testloader.dataset.targets[positive_sample_index]
#     new_loader = copy.deepcopy(src_testloader)
    
#     # src_index = np.array([1,2,3,4,5,101,102,103,104,105,201,202,203,204,205,301,302,303,304,305,401,402,403,404,405,501,502,503,504,505,601,602,603,604,605,701,702,703,704,705,801,802,803,804,805,901,902,903,904,905])

#     new_dataset = new_loader.dataset
#     new_dataset.data = np.concatenate((src_testloader.dataset.data[src_index], positive_sample_data))
#     new_dataset.targets = np.concatenate((src_testloader.dataset.targets[src_index], positive_sample_targets))
#     # new_dataset.data = positive_sample_data
#     # new_dataset.targets = positive_sample_targets

#     new_dataloader = torch.utils.data.DataLoader(dataset=new_dataset, batch_size=100, shuffle=True, num_workers=0,pin_memory=True)
    
#     # print(new_dataloader.dataset.data.shape)
#     # print(new_dataloader.dataset.targets.shape)
#     return new_dataloader  

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
