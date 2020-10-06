# coding: utf-8
import pickle
import numpy as np
import os,random
import keras
import matplotlib.pyplot as plt
from utils.channel_pruning import set_compact_model_weights,freeze_build_cap
from utils.evalute import plot_and_evalute
os.environ["KERAS_BACKEND"] = "tensorflow"
from model.resnet import resnet
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

file_orig = 'model/ResNet_Model.h5'#存放原始模型
file_compact='model/ResNet_Model_comp.h5'#存放剪枝后的模型

sparse_factor=1e-3#l1稀疏因子
percent=0.5#要裁掉的的参数比例
learning_rate=0.0001#学习率
batch_size=1024
nb_epoch = 50     # number of epochs to train on

#Load Data
Xd = pickle.load(open("./data/RML2016.10a/RML2016.10a_dict.pkl",'rb'),encoding='latin-1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
X_N=[]
lab = []
M_old=['AM-SSB','WBFM','GFSK','BPSK','QPSK']#只选取5类进行训练预测
M_new=list(set(mods)-set(M_old))

for mod in M_old:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):
            lab.append((mod,snr))
X=np.vstack(X)

# for mod in M_new:
#     for snr in snrs:
#         X_N.append(Xd[(mod,snr)])
# X_N=np.vstack(X_N)

#=====================================================分割已知类型数据为训练集和测试集====================================
np.random.seed(2019)
X=X.transpose(0,2,1)
n_examples=X.shape[0]
n_train=n_examples*0.8#8:1:1分训练，验证，测试数据
n_val=n_examples*0.1
train_idx=np.random.choice(range(0,n_examples),size=int(n_train),replace=False)
rest=np.array(list(set(range(0,n_examples))-set(train_idx)))
test_idx=rest[np.random.choice(range(0,len(rest)),size=int(n_val),replace=False)]
val_idx=np.array([item for item in list(set(rest)-set(test_idx)) if lab[item][1]>=10])#验证集只关注SNR大于5的情况，信噪比太低的数据网络不具备学习能力
np.random.shuffle(train_idx)
X_train=X[train_idx]
X_test=X[test_idx]
X_val=X[val_idx]

def one_hot(x):
    xx=np.zeros([len(x),max(x)+1])
    xx[np.arange(len(x)),x]=1
    return xx

Y_train=one_hot(list(map(lambda x:M_old.index(lab[x][0]),train_idx)))#构造one-hot label
Y_test=one_hot(list(map(lambda x:M_old.index(lab[x][0]),test_idx)))
Y_val=one_hot(list(map(lambda x:M_old.index(lab[x][0]),val_idx)))

print(X_train.shape,Y_train.shape)
in_shape=X.shape

#========================================================训练模型=======================================================
flag=1
cap=[]
"""搭建模型进行训练"""
while flag==1:
    model = resnet(in_shape, len(M_old), sparse_factor, cap=[])
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=learning_rate/nb_epoch)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    for _ in range(2):
        history=model.fit(X_train,
            Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=2,
            validation_data=(X_val, Y_val),
            callbacks = [
                keras.callbacks.ModelCheckpoint(file_orig, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
            ])

    # Evaluate and Plot Model Performance
    plot_and_evalute('origin_model', model, X_test, Y_test, history, snrs, lab, test_idx, M_old)
    cap,cap_mask=freeze_build_cap(model,percent)

    if (0 in cap):flag=1 #每个layer至少得存在两个channel，若剪枝完某个通道channel数量为0了，则重新训练直到符合条件.
    else:flag=0

model.summary()
# =================================================开始剪枝=============================================================
compact_model = resnet(in_shape, len(M_old),sparse_factor,cap=cap)#根据cap参数预创建一个剪枝后的模型
compact_model.summary()
set_compact_model_weights(model, compact_model,cap_mask)

# =================================================fine_tune============================================================
adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=learning_rate/nb_epoch)
compact_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#打乱数据再训练
train_idx=np.random.choice(range(0,X_train.shape[0]),size=int(n_train),replace=False)
X_train=X_train[train_idx]
Y_train=Y_train[train_idx]

history=compact_model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          nb_epoch=50,
          verbose=2,
          validation_data=(X_val, Y_val),
          callbacks=[
              keras.callbacks.ModelCheckpoint(file_compact, monitor='val_loss', verbose=0, save_best_only=True,
                                              mode='auto'),
              keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
          ])
#Evaluate and Plot Model Performance
plot_and_evalute('compact_model',compact_model,X_test,Y_test,history,snrs,lab,test_idx,M_old)










