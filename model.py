from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Deconv2D, Lambda
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

def downsampling_conv_block(net, filters, kernel=3):#U-net下采样部分 
    net = Conv2D(filters, kernel, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv2D(filters, kernel, padding='same', activation='relu')(net)
    _net = BatchNormalization()(net)
    net = MaxPooling2D()(_net)
    return net, _net#_net是为了后面的跳跃连接

# keras 搭建网络时，只可以使用keras类型的变量，若要用一些tf的操作则需考虑使用Lambda层 如下所示，其中K为调用keras的底层，常用如tf、theano、mxnet等
def upsampling_conv_block(net1, net2, filters, kernel=3):
    net1 = Deconv2D(filters, kernel, strides=2, padding='same', activation='relu')(net1)
    net = Lambda(lambda x: K.concatenate(x, -1))([net1, net2])
    net = Conv2D(filters, kernel, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv2D(filters, kernel, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    return net

# 在搭建工程时，网络常常用类搭建，本程序为自娱自乐，用函数节省时间
def unet(istraining=True):
    input = Input(shape=(256, 512, 1))
    net1, _net1 = downsampling_conv_block(input, 32)
    net2, _net2 = downsampling_conv_block(net1, 64)
    net3, _net3 = downsampling_conv_block(net2, 128)
    net4, _net4 = downsampling_conv_block(net3, 256)
    net5 = Conv2D(512, 3, padding='same', activation='relu')(net4)
    net5 = BatchNormalization()(net5)
    net5 = Conv2D(256, 1, activation='relu')(net5)
    net5 = BatchNormalization()(net5)
    net5 = Conv2D(256, 3, padding='same', activation='relu')(net5)
    net5 = BatchNormalization()(net5)
    net5 = Conv2D(512, 1, activation='relu')(net5)
    net5 = BatchNormalization()(net5)
    net6 = upsampling_conv_block(net5, _net4, 256)
    net7 = upsampling_conv_block(net6, _net3, 128)
    net8 = upsampling_conv_block(net7, _net2, 64)
    net9 = upsampling_conv_block(net8, _net1, 32)
    output = Conv2D(4, 1, padding='same', activation='sigmoid')(net9)
    model = Model(inputs=input, outputs=output)
    if istraining:
        compile_model(model)
    return model

def focal_loss(gamma=2.):
    def focal_loss_fixed(y_true, y_pred):
        eps = 0.0001
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))#预测值和真实值都是1
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))#预测值和真实值都是0
        return -K.mean(K.pow(1 - pt_1, gamma) * K.log(pt_1 + eps)) - K.mean(K.pow(pt_0, gamma) * K.log(1 - pt_0 + eps))
    return focal_loss_fixed


def dice_coefficient(y_pre, y_true):
    eps = 0.0001
    y_pre = tf.where(y_pre > 0.5, K.ones_like(y_pre), K.zeros_like(y_pre))
    # int_sec = K.sum(y_pre * y_true, [1, 2, 3])
    # xy_sum = K.sum(y_true, [1, 2, 3]) + K.sum(y_pre, [1, 2, 3])
    int_sec = K.sum(y_pre * y_true)
    xy_sum = K.sum(y_true) + K.sum(y_pre)
    return (2 * int_sec + eps) / (xy_sum + eps)

# loss可以使用keras.losses里自带的loss，也可以如上自己定义，但是损失函数默认两个参数必须为y_true和y_pred，若欲使用其他参数
# 可以如上focal_loss函数嵌套的方法调用，metrics也同理
def compile_model(model):
    opt = Adam(lr=0.0005)
    model.compile(opt, focal_loss(), metrics=[dice_coefficient])



