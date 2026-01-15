import torch
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as F
min_var_est = 1e-8



def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    source = source.reshape(source.size()[0],source.size()[1])
    target = target.reshape(target.size()[0],target.size()[1])
    
    n_samples = int(source.size()[0])+int(target.size()[0])
    
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)
 

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
  
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list ] #kernel_num*n_samples*n_samples
    
    

    return sum(kernel_val)#/len(kernel_val) #n_samples*n_samples


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size] # Source<->Source
    YY = kernels[batch_size:, batch_size:] # Target<->Target
    XY = kernels[:batch_size, batch_size:] # Source<->Target
    YX = kernels[batch_size:, :batch_size] # Target<->Source

    loss = torch.mean(XX + YY - XY -YX) 

   
    return loss

def ammd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size_source = int(source.size()[0]) 
    batch_size_target = int(target.size()[0]) 

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size_source, :batch_size_source] # Source<->Source
    YY = kernels[batch_size_source:, batch_size_source:] # Target<->Target
    XY = kernels[:batch_size_source, batch_size_source:] # Source<->Target
    YX = kernels[batch_size_source:, :batch_size_source] # Target<->Source

    loss = torch.sum(XX)/batch_size_source**2 + torch.sum(YY)/batch_size_target**2  - torch.sum(XY)/(batch_size_source*batch_size_target) - torch.sum(YX)/(batch_size_source*batch_size_target)
   
    return loss

def cmmd(source, target, s_label, t_label,num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])

    
    s_label = s_label.cpu()
    s_label = s_label.view(batch_size,1)
    s_label = torch.zeros(batch_size,num_class).scatter_(1, s_label.data, 1)
    s_label = Variable(s_label).cuda()

    t_label = t_label.cpu()
    t_label = t_label.view(batch_size, 1)
    t_label = torch.zeros(batch_size,num_class).scatter_(1, t_label.data, 1)
    t_label = Variable(t_label).cuda()


    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    loss += torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX +
                      torch.mm(t_label, torch.transpose(t_label, 0, 1)) * YY -
                      2 * torch.mm(s_label, torch.transpose(t_label, 0, 1)) * XY)
    return loss


def cal_weight(source_label, target_logits, num_class):
    batch_size = source_label.size()[0]
    source_label = source_label.cpu().data.numpy()
    source_label_onehot = np.eye(num_class)[source_label] # one hot

    source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, num_class)
    source_label_sum[source_label_sum == 0] = 100
    source_label_onehot = source_label_onehot / source_label_sum # label ratio

    # Pseudo label
    target_label = target_logits.cpu().data.max(1)[1].numpy()

    target_logits = target_logits.cpu().data.numpy()
    target_logits_sum = np.sum(target_logits, axis=0).reshape(1, num_class)
    target_logits_sum[target_logits_sum == 0] = 100
    target_logits = target_logits / target_logits_sum

    weight_ss = np.zeros((batch_size, batch_size))
    weight_tt = np.zeros((batch_size, batch_size))
    weight_st = np.zeros((batch_size, batch_size))

    set_s = set(source_label)
    set_t = set(target_label)
    count = 0
    for i in range(num_class): # (B, C)
        if i in set_s and i in set_t:
            s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
            t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)
            
            ss = np.dot(s_tvec, s_tvec.T) # (B, B)
            weight_ss = weight_ss + ss
            tt = np.dot(t_tvec, t_tvec.T)
            weight_tt = weight_tt + tt
            st = np.dot(s_tvec, t_tvec.T)
            weight_st = weight_st + st     
            count += 1

    length = count
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])
    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


    
def lmmd(source, target, s_label, t_label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None):


    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = cal_weight(s_label,t_label, num_class)
    weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )

    return loss



def Supervised_InfoNCE_loss(features, labels, temperature=0.07):
    '''
    The InfoNCE loss for supervised learning
    features: (batch_size, feature_dim)
    labels: (batch_size)
    temperature: default 0.07
    '''
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.t())
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

    pos_pair_sim = similarity_matrix * labels_matrix.float()
    pos_pair_sim = torch.exp(pos_pair_sim / temperature)

    neg_pair_sim = similarity_matrix * (~labels_matrix).float()
    neg_pair_sim = torch.exp(neg_pair_sim / temperature)

    contrastive_loss = -torch.log(pos_pair_sim / (pos_pair_sim + torch.sum(neg_pair_sim, dim=1)))
    contrastive_loss = torch.mean(contrastive_loss)

    return contrastive_loss

def P_contrastive_loss(prototypes, features, labels, temperature=0.07):
    '''
    The Prototypes-based InfoNCE Loss
    '''
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.cdist(features, prototypes, p=2)
    labels_matrix = labels.unsqueeze(1) == torch.arange(prototypes.size(0)).unsqueeze(0)

    pos_pair_sim = similarity_matrix * labels_matrix.float()
    pos_pair_sim = torch.exp(pos_pair_sim / temperature)

    neg_pair_sim = similarity_matrix * (~labels_matrix).float()
    neg_pair_sim = torch.exp(neg_pair_sim / temperature)
    
    contrastive_loss = -torch.log(pos_pair_sim / (pos_pair_sim + torch.sum(neg_pair_sim, dim=1)))
    contrastive_loss = torch.mean(contrastive_loss)

    return contrastive_loss

def contrastive_loss(src_features,tgt_features):

    src_features = F.normalize(src_features, dim=1)
    tgt_features = F.normalize(tgt_features, dim=1)

    similarity_matrix = torch.matmul(src_features, tgt_features.t())
    loss = torch.mean(similarity_matrix)


    return loss








def mix_rbf_mmsd(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmsd(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = torch.matmul(X, X.t())
    XY = torch.matmul(X, Y.t())
    YY = torch.matmul(Y, Y.t())

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    r = lambda x: x.unsqueeze(0)
    c = lambda x: x.unsqueeze(1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
    return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))


def _mmsd(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)
    n = K_YY.size(0)
    
    C_K_XX = torch.pow(K_XX, 2)
    C_K_YY = torch.pow(K_YY, 2)
    C_K_XY = torch.pow(K_XY, 2)

    if biased:
        mmsd = (torch.sum(C_K_XX) / (m * m)
              + torch.sum(C_K_YY) / (n * n)
              - 2 * torch.sum(C_K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = torch.trace(C_K_XX)
            trace_Y = torch.trace(C_K_YY)

        mmsd = ((torch.sum(C_K_XX) - trace_X) / ((m-1) * m)
              + (torch.sum(C_K_YY) - trace_Y) / ((n-1) * n)
              - 2 * torch.sum(C_K_XY) / (m * n))
    return mmsd


def mmsd_rbf_loss(X1, X2, bandwidths=[3]):
    kernel_loss = mix_rbf_mmsd(X1, X2, sigmas=bandwidths)
    return kernel_loss


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    entropy = torch.mean(entropy)
    return entropy 

def DivEntropy(input_):
    epsilon = 1e-5
    msoftmax = input_.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + epsilon))
    return gentropy_loss