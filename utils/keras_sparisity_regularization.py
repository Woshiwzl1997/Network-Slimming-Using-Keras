"""
添加稀疏正则化层来剪枝，但论文不建议这么做，故弃之
"""

from keras.engine.topology import Layer #自定义层
from keras.engine.topology import InputSpec
import keras.backend as K
from keras import initializers
from keras import regularizers
from keras.regularizers import l1
import tensorflow as tf

class SparsityRegularization(Layer):
    def __init__(self, l1=0.01, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = -1#通道所在维度
        else:
            self.axis = 1
        self.l1 = l1
        super(SparsityRegularization, self).__init__(**kwargs)

    def build(self, input_shape):#定义自定义权重
        dim = input_shape[self.axis]#通道数
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),#指定输入layer是几维，类型和shape
                                    axes={self.axis: dim})#进行操作的那一维的大小，这里是在channel维[-1]进行操作
        self.gamma = self.add_weight(shape=(dim,),
                                     initializer=initializers.get('ones'),
                                     name='gamma',
                                     regularizer=regularizers.get(l1(l=self.l1)),
                                     trainable=True
                                     )
        self.trainable_weights = [self.gamma]
        super(SparsityRegularization, self).build(input_shape)#标配，写上就完事了

    def call(self, inputs, mask=None):#编写层的功能逻辑，input为输入张量
        return inputs * self.gamma

    def compute_output_shape(self, input_shape):#定义形状变化逻辑
        return input_shape

    def get_config(self):
        config = {
            'L1': self.l1
        }
        base_config = super(SparsityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
