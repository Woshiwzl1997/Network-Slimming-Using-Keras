from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,MaxPooling2D
from utils.keras_sparisity_regularization import SparsityRegularization
from copy import deepcopy
import numpy as np
import re

def is_int(x):
    """
    判断x是否为整数
    """
    x=abs(x)
    if x-int(x)>0.0:
        return False
    else:
        return True

def set_compact_model_weights(origin_model, compact_model, cap_mask):
    bn_id_in_cap=0#第几个BN
    cov_id_in_cap=0#第几个conv2d
    k_id_in=int(0)#第几个keepdim
    k_id_in_cap=-1#keepdim层卷积核的输入维度对应的 cap index
    k_id_out_cap=0#keepdim层卷积核的输出维度对应的 cap index
    for o, c in zip(origin_model.layers, compact_model.layers):
        if isinstance(o, BatchNormalization):
            # get not zero value index
            idx = np.squeeze(np.argwhere(cap_mask[bn_id_in_cap] != 0))
            gamma=o.get_weights()[0][idx]  # gamma
            beta=o.get_weights()[1][idx] # beta
            mean=o.get_weights()[2][idx]  # mean
            var=o.get_weights()[3][idx]
            if idx.size == 1: #维度必须为一维
                gamma = np.expand_dims(gamma, 0)
                beta = np.expand_dims(beta, 0)
                mean = np.expand_dims(mean, 0)
                var = np.expand_dims(var, 0)
            c.set_weights([deepcopy(gamma),#gamma
                           deepcopy(beta),#beta
                           deepcopy(mean),#mean
                           deepcopy(var)])#var
            bn_id_in_cap +=1

        elif isinstance(o,Conv2D) and re.search('_(.*)\d',o.name).group(1)!='keepdim':#如果Conv2D不是用来keepdim
            if cov_id_in_cap==0:#第一个卷积核上一层为数据输入，只变卷积核输出维度
                idx_out=np.squeeze(np.argwhere(cap_mask[cov_id_in_cap] != 0))

                w=o.get_weights()[0][:,:,:,idx_out]
                if idx_out.size==1:w=np.expand_dims(w,3)

                b = o.get_weights()[1][idx_out]
                if idx_out.size == 1: b = np.expand_dims(b, 0)

                c.set_weights([deepcopy(w),  # conv weight
                               deepcopy(b)  # bias
                               ])

            elif is_int((cov_id_in_cap-1)/5) or is_int((cov_id_in_cap-3)/5) or is_int((cov_id_in_cap-5)/5):#除以1,3,5为商为整数的情况
                idx_in=np.squeeze(np.argwhere(cap_mask[int((cov_id_in_cap-1)/5)*5] != 0))
                idx_out=np.squeeze(np.argwhere(cap_mask[cov_id_in_cap] != 0))

                w = o.get_weights()[0][:, :, :, idx_out]#分两次取维度
                if idx_out.size == 1: w = np.expand_dims(w, 3)#为了防止变成三维，需要reshape

                w=w[:,:,idx_in,:]
                if idx_in.size == 1: w = np.expand_dims(w, 2)

                b=o.get_weights()[1][idx_out]
                if idx_out.size == 1: b = np.expand_dims(b, 0)

                c.set_weights([deepcopy(w),  # conv weight
                               deepcopy(b)# bias
                               ])
            else:
                idx_in = np.squeeze(np.argwhere(cap_mask[(cov_id_in_cap - 1)] != 0))
                idx_out = np.squeeze(np.argwhere(cap_mask[cov_id_in_cap] != 0))

                w = o.get_weights()[0][:, :, :, idx_out]
                if idx_out.size == 1: w = np.expand_dims(w, 3)

                w = w[:, :, idx_in, :]
                if idx_in.size == 1: w = np.expand_dims(w, 2)

                b = o.get_weights()[1][idx_out]
                if idx_out.size == 1: b = np.expand_dims(b, 0)

                c.set_weights([deepcopy(w),  # conv weight
                               deepcopy(b)  # bias
                               ])
            cov_id_in_cap += 1
        elif isinstance(o,Conv2D) and re.search('_(.*)\d',o.name).group(1)=='keepdim':#如果是keepdim卷积层
            if k_id_in%2 == 0:#偶数加3
                k_id_in_cap += 3
            elif k_id_in%2==1:#奇数加二
                k_id_in_cap +=2
            idx_in=np.squeeze(np.argwhere(cap_mask[k_id_in_cap] != 0))
            idx_out=np.squeeze(np.argwhere(cap_mask[int(k_id_in/2)*5] != 0))

            w = o.get_weights()[0][:, :, :, idx_out]
            if idx_out.size == 1: w = np.expand_dims(w, 3)

            w = w[:, :, idx_in, :]
            if idx_in.size == 1: w = np.expand_dims(w, 2)

            b = o.get_weights()[1][idx_out]
            if idx_out.size == 1: b = np.expand_dims(b, 0)

            c.set_weights([deepcopy(w),  # conv weight
                          deepcopy(b)  # bias
                          ])
            k_id_out_cap += 5
            k_id_in += 1



"""
根据剪枝比例，遍历整个网络，统计每层需要剪去的通道,并将其置零
cap：channel after pruning
"""
def freeze_build_cap(model,percent):
    """
    :param model: 模型
    :param percent: 剪枝比例
    :return: 每层剪枝过后的通道：cap
    """
    total=0#整个网络的特征数目之和
    for m in model.layers:
        if isinstance(m,BatchNormalization):#如果发现BN层
            total += m.get_weights()[0].shape[0]#BN.get_wrights():获得0：gamma,1：beta,2:moving_mean,3:moving_variance

    bn=np.zeros(total)
    index=0
    for m in model.layers:
        if isinstance(m, BatchNormalization):
            size=m.get_weights()[0].shape[0]
            bn[index:(index+size)]=np.abs(deepcopy(m.get_weights()[0]))# 把所有BN层gamma值拷贝下来
            index+=size

    #根据所有BN层的权重确定剪枝比例
    y=np.sort(bn)#将网络所有BN层的权重排序
    thre_index=int(total*percent)#确顶要保留的参数大小
    thre=y[thre_index]#最小的权重值

    pruned=np.array(0.0)
    cap=[]#确定每个BN层要保留的参数数目
    cap_mask=[]#每个BN层要保留的参数MASK
    for k,m in enumerate(model.layers):
        if isinstance(m,BatchNormalization):
            weight_copy=deepcopy(m.get_weights()[0])
            mask=np.array([1.0 if item>thre else 0.0 for item in weight_copy])# 小于权重thre的为0，大于的为1,唯独保持不变
            pruned=pruned + mask.shape[0] -np.sum(mask)# 小于权重thre的为0，大于的为1,唯独保持不变
            m.set_weights([                   #貌似只能一起赋值
                m.get_weights()[0] * mask,
                m.get_weights()[1] * mask,
                m.get_weights()[2],
                m.get_weights()[3]
            ])
            cap.append(int(np.sum(mask)))
            cap_mask.append(deepcopy(mask))
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(np.sum(mask))))
    print('Pre-processing Successful!')

    print("Num of layer after pruning):")
    print(cap)
    return cap, cap_mask



