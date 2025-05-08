import torch
import math
import numpy as np
import os
import pickle
from torch import bmm
from model.neighborCorrelation import NeighborCorrelation

from torch.autograd import Variable


def cal_similarity(object_node_feat, neigh_node_feat, neigh_nodes, layers):
    '''
    :param object_node_feat: 目标结点的特征
    :param neigh_node_feat:  目标结点的所有邻居结点特征
    :param probability:      取样概率 其实这个参数可以由下面的layers得到 p = (1/2)的j-1次方
    :param layers:           图神经网络的第n层，
    :param neigh_nodes:      目标结点的邻居数量，
    :return:                 返回的是一个前k个相似度索引，以方便进行下一次的聚合操作
    '''
    # object_node_feat.cuda()
    # neigh_node_feat.cuda()
    similarity = []  # 用于保存目标结点与邻居结点的相似度
    p = pow(1 / 2, layers)  # 取样概率
    sample_nums = math.ceil(neigh_nodes * p)  # 取样数量  不加取样的那一行直接换成0就行  neigh_nodes就是邻居数量，也就是目标结点的度
    for i in range(neigh_nodes):
        sim = torch.mul(object_node_feat, neigh_node_feat[i]).sum(dim=0)
        sim = sim.detach().cpu()
        similarity.append(sim)
    # 返回前n个大的值
    index = np.argsort(similarity)[-sample_nums:].tolist()
    return index, sample_nums

'''自适应取样代码'''
def adaptive_sampling(object_node_feat, neigh_node_feat):
   '''
   :param object_node_feat: 目标节点的特征，注意这是一个1*dim的张量
   :param neigh_node_feat: 邻居节点特征，这时一个n*dim的张量
   我们应该首先要做的就是把目标节点的特征扩展成和邻居节点特征相同的维度，然后利用神经网络进行特征变换，然后计算相似度
   最终计算出来的相似度应该是一个n*1的张量，我们需要对这个张量进行求平均值，还得逐个将n*1当中的每一个元素与这个平均值进行比较
   只有大于这个平均值的具体特征，才参与特征的聚合操作，这个地方的返回值主要是特征索引，也就是特征的标号，这样我们只需要根据这个标号对原来的特征进行
   特征聚合操作就完事了。但是实际在操作的时候要注意调试，别把特征弄错了
   这一块只要修改完了，后面基本上都一样了，直接把原来的代码放过来简单修改一下就好啦
   :return:
   '''

   input_dim = 128
   output_dim = 256
   '''这个地方的代码放到服务器上时，注意要在后面加上.to('cuda')'''
   model = NeighborCorrelation(input_dim, output_dim)  #进行模型的初始化工作
   object_node_feat_transformed = model(object_node_feat) #进行目标节点特征的线性变换操作
   neigh_node_feat_transformed = model(neigh_node_feat)  #进行邻居节点特征的线性变换操作

   #将object_node_feat_transformed扩展成与neigh_node_feat_transformed相同特征维度的一个矩阵 M1
   M1 = object_node_feat_transformed.expand(neigh_node_feat_transformed.shape[0], -1)  # m x 1024
   # 将 M1 与 neigh_node_feat_transformed按行相乘得到 H4，并计算每行元素的和
   H4 = torch.mul(M1, neigh_node_feat_transformed)  # m x 1024
   H4_row_sum = torch.sum(H4, dim=1)  # m
   #计算 H4_row_sum 中所有元素的平均值，得到平均相关度 avg
   avg = torch.mean(H4_row_sum)

   #然后进行取样操作，需要根据计算出来的张量H4_row_sum当中的每一个元素与avg进行比较，如果
   #该张量中的元素值大于avg，那么我将其进行保留，并返回对应的索引
   # 标记大于 avg 的元素位置
   marked_indices = torch.nonzero(H4_row_sum > avg)
   '''
   这个地方需要注意，不知道后面模型的聚合操作需不需要判断一下marked_indices的维度大小，
   然后根据具体的维度信息来进行获取对应位置的索引
   '''
   # 返回标记的位置
   marked_indices = marked_indices.squeeze(1)
   # 将张量marked_indices转换为列表
   tensor_list = marked_indices.tolist()
   return tensor_list , len(tensor_list)  #将元素索引以列表形式进行返回，并将列表的长度进行返回，用来判断一共取样了多少个邻居节点


def load_visual_text(path_visual):
   with open(path_visual,'rb') as fr:
      visual_feat = pickle.load(fr)
   return visual_feat


# 遍历文件夹下的文件,直接返回其列表就行，到时候直接引用 0：user_clo 1:user_user 2:clo_user 3:clo_clo
def traversal(path):
    Filelist = []
    graph_list = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    # 然后再对这个文件列表进行遍历，读取出全部的交互数据图
    print(Filelist)
    for i in Filelist:
        with open(i, 'rb') as fr:
            graph_list.append(pickle.load(fr))
    return graph_list


def overfit(Us, Is, Js, Ks, model):
    return model.fit(set(Us), set(Is + Js + Ks))

#对比损失函数定义
def constrasive_loss(score1,score2):
    '''
    :param score1: 正例样本分数  这就是一个张量
    :param score2: 负例样本分数
    :return: 返回正例样本经过归一化之后的分数
    '''
    temperature = 0.8  # 这是一个超参数
    exp_score1 = torch.exp(score1 / temperature)
    exp_score2 = torch.exp(score2 / temperature)
    exp_scores = exp_score1 + exp_score2
    sum_exp_scores = torch.sum(exp_scores)
    softmax_probs = score1 / sum_exp_scores
    log_softmax_probs = (-torch.log(softmax_probs)).sum()
    return log_softmax_probs


def bpr(data, model, v_feat, t_feat, mode='train', emb_dim=128, model2=None):
    Us = [int(pair[0]) for pair in data]  # 目标用户
    Is = [int(pair[1]) for pair in data]  # 上衣结点
    Js = [int(pair[2]) for pair in data]  # 下衣正例结点
    Ks = [int(pair[3]) for pair in data]  # 下衣负例结点

    f_u_v_from_uc, f_u_t_from_uc = model.forward_u(Us, idx=2, mode=mode, v_feat=v_feat, t_feat=t_feat)
    f_u_v_from_uu, f_u_t_from_uu = model.forward_u(Us, idx=3, mode=mode)
    f_u_v = 0.6 * f_u_v_from_uc + 0.4 * f_u_v_from_uu
    f_u_t = 0.6 * f_u_t_from_uc + 0.4 * f_u_t_from_uu

    #f_u_v = f_u_v_from_uc
    #f_u_t = f_u_t_from_uc

    f_I_v_feat_cc, f_I_t_feat_cc, f_J_v_feat_cc, \
    f_J_t_feat_cc, f_K_v_feat_cc, f_K_t_feat_cc = model.forward_c(Is, Js, Ks, idx=0, mode=mode, v_feat=v_feat, t_feat=t_feat)

    f_I_v_feat_cu, f_I_t_feat_cu, f_J_v_feat_cu, \
    f_J_t_feat_cu, f_K_v_feat_cu, f_K_t_feat_cu = model.forward_c(Is, Js, Ks, idx=1, mode=mode, v_feat=v_feat, t_feat=t_feat)

    f_top_v = 0.8 * f_I_v_feat_cu + 0.2 * f_I_v_feat_cc
    f_top_t = 0.5 * f_I_t_feat_cu + 0.5 * f_I_t_feat_cc

    f_pob_v = 0.8 * f_J_v_feat_cu + 0.2 * f_J_v_feat_cc
    f_pob_t = 0.5 * f_J_t_feat_cu + 0.5 * f_J_t_feat_cc

    f_pab_v = 0.8 * f_K_v_feat_cu + 0.2 * f_K_v_feat_cc
    f_pab_t = 0.5 * f_K_t_feat_cu + 0.5 * f_K_t_feat_cc

    visual_ij = bmm(f_top_v.unsqueeze(1), f_pob_v.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    text_ij = bmm(f_top_t.unsqueeze(1), f_pob_t.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    visual_ik = bmm(f_top_v.unsqueeze(1), f_pab_v.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    text_ik = bmm(f_top_t.unsqueeze(1), f_pab_t.unsqueeze(-1)).squeeze(-1).squeeze(-1)


    p_ij = 0.5 * visual_ij + 0.5 * text_ij
    p_ik = 0.5 * visual_ik + 0.5 * text_ik


    batchsize = len(f_u_v)  # 注意要把下面的512改成向量拼接之后的维度
    cuj = bmm(f_u_v.view(batchsize, 1, emb_dim), f_pob_v.view(batchsize, emb_dim, 1)).view(batchsize) + \
          bmm(f_u_t.view(batchsize, 1, emb_dim), f_pob_t.view(batchsize, emb_dim, 1)).view(batchsize)
    cuk = bmm(f_u_v.view(batchsize, 1, emb_dim), f_pab_v.view(batchsize, emb_dim, 1)).view(batchsize) + \
          bmm(f_u_t.view(batchsize, 1, emb_dim), f_pab_t.view(batchsize, emb_dim, 1)).view(batchsize)

    score1 = 0.3 * p_ij + 0.7 * cuj  # 正例样本分数
    score2 = 0.3 * p_ik + 0.7 * cuk  # 负例样本分数
    #在这个地方加上对比损失函数
    log_softmax_probs = constrasive_loss(score1,score2)


    # 增加一个判断 如果是train方法，就增加返回正则项那一项，如果是验证测试方法，就不返回正则化的那一项
    if mode == 'train':
        # 定义一个方法  讲，各个服装I，J，K以及用户传入到此
        cukjweight = overfit(Us, Is, Js, Ks, model2)
        return score1 - score2, cukjweight, log_softmax_probs
    if mode == 'valid':
        return score1 - score2