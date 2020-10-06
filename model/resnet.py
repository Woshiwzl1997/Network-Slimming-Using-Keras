# coding: utf-8
import os,random
from keras.layers import Input,Reshape,ZeroPadding2D,Conv2D,Dropout,Flatten,Dense,Activation,MaxPooling2D,AlphaDropout
from keras import layers
import keras.models as Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1


os.environ["KERAS_BACKEND"] = "tensorflow"

#创建模型
def residual_stack(Xm,kennel_size,Seq,pool_size,sparse_facor,cap=[]):
    if cap==[]:
        #1*1 Conv Linear
        Xm = Conv2D(32, (1, 1), padding='same', name=Seq+"_conv1", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch1",gamma_regularizer=l1(sparse_facor))(Xm)#在通道维进行BN，对gamma稀疏性进行约束
        Xm = Activation('relu')(Xm)
        #Residual Unit 1
        Xm_shortcut = Xm
        Xm = Conv2D(32, kennel_size, padding='same',name=Seq+"_conv2", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch2",gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv3", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch3",gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(int(Xm_shortcut.shape[3]), kennel_size, padding='same', name=Seq + "_keepdim1",kernel_initializer='glorot_normal')(Xm)#保持与shortcut相同的维度
        Xm = layers.add([Xm,Xm_shortcut])
        Xm = Activation("relu")(Xm)
        #Residual Unit 2
        Xm_shortcut = Xm
        Xm = Conv2D(32, kennel_size, padding='same',name=Seq+"_conv4", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch4",gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv5", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch5",gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(int(Xm_shortcut.shape[3]), kennel_size, padding='same', name=Seq + "_keepdim2",kernel_initializer='glorot_normal')(Xm)
        Xm = layers.add([Xm,Xm_shortcut])
        Xm = Activation("relu")(Xm)
    else:
        # 1*1 Conv Linear
        Xm = Conv2D(cap[0], (1, 1), padding='same', name=Seq + "_conv1", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch1", gamma_regularizer=l1(sparse_facor))(Xm)  # 在通道维进行BN，对gamma稀疏性进行约束
        Xm = Activation('relu')(Xm)
        # Residual Unit 1
        Xm_shortcut = Xm
        Xm = Conv2D(cap[1], kennel_size, padding='same', name=Seq + "_conv2", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch2", gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(cap[2], kennel_size, padding='same', name=Seq + "_conv3", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch3", gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(int(Xm_shortcut.shape[3]), kennel_size, padding='same', name=Seq + "_keepdim1",kernel_initializer='glorot_normal')(Xm)
        Xm = layers.add([Xm, Xm_shortcut])
        Xm = Activation("relu")(Xm)
        # Residual Unit 2
        Xm_shortcut = Xm
        Xm = Conv2D(cap[3], kennel_size, padding='same', name=Seq + "_conv4", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch4", gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(cap[4], kennel_size, padding='same', name=Seq + "_conv5", kernel_initializer='glorot_normal')(Xm)
        Xm = BatchNormalization(axis=-1, name=Seq + "batch5", gamma_regularizer=l1(sparse_facor))(Xm)
        Xm = Activation("relu")(Xm)
        Xm = Conv2D(int(Xm_shortcut.shape[3]),kennel_size, padding='same', name=Seq + "_keepdim2", kernel_initializer='glorot_normal')(Xm)
        Xm = layers.add([Xm, Xm_shortcut])
        Xm = Activation("relu")(Xm)
    #MaxPooling
    Xm = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid')(Xm)
    return Xm


def resnet(input_shape,num_cat,sparse_factor,cap):
    """
    :param X_shape: 输入维度
    :param num_cat: 输出维度
    :param cap:剪枝后模型的参数
    :return:Model
    """
    Xm_input=Input(input_shape[1:])
    Xm=Reshape([input_shape[1],input_shape[2],1],input_shape=input_shape[1:])(Xm_input)
    if cap == []:
        Xm = residual_stack(Xm,kennel_size=(5,2),Seq="ReStk0",pool_size=(2,2),sparse_facor=sparse_factor,cap=[])   #shape:(64,1,32)
        Xm = residual_stack(Xm,kennel_size=(5,1),Seq="ReStk1",pool_size=(2,1),sparse_facor=sparse_factor,cap=[])   #shape:(32,1,32)
        Xm = residual_stack(Xm,kennel_size=(5,1),Seq="ReStk2",pool_size=(2,1),sparse_facor=sparse_factor,cap=[])   #shape:(16,1,32)
    else:
        Xm = residual_stack(Xm,kennel_size=(5,2),Seq="ReStk0",pool_size=(2,2),sparse_facor=sparse_factor,cap=cap[0:5])#每个res_stack有5个BN层
        Xm = residual_stack(Xm,kennel_size=(5,1),Seq="ReStk1",pool_size=(2,1),sparse_facor=sparse_factor,cap=cap[5:10])
        Xm = residual_stack(Xm,kennel_size=(5,1),Seq="ReStk2",pool_size=(2,1),sparse_facor=sparse_factor,cap=cap[10:15])
    Xm = Flatten()(Xm)
    Xm = Dense(32, activation='selu', kernel_initializer='glorot_normal', name="dense1")(Xm)
    Xm = AlphaDropout(0.3)(Xm)
    Xm = Dense(16, activation='selu', kernel_initializer='glorot_normal', name="dense2")(Xm)
    Xm = AlphaDropout(0.3)(Xm)
    #Full Con 2
    Xm = Dense(num_cat, kernel_initializer='glorot_normal', name="dense3")(Xm)
    #SoftMax
    Xm = Activation('softmax')(Xm)
    #Create Model
    model = Model.Model(inputs=Xm_input, outputs=Xm)
    return model