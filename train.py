from model import *
from data import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

epochs = 30
batchsize = 16
is_mining_train = True

path = '../severstal-steel-defect-detection/train.csv'
data, label = read_data(path)
if is_mining_train:
    #data, _, label, _ = split_nan(data, label)#这里只取了有缺陷的做训练
    data,label=split_all(data,label)#取所有的数据进行第一次的测试
#train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
td, vd, tl, vl = train_test_split(data, label, test_size=0.2)#td,vd用于训练，20%的tl和vl用于测试
tg = generator(td, tl, batchsize)
vg = generator(vd, vl, batchsize)
net = unet()

#net.load_weights('./base_models/ver1.hdf5')

save_path = './models/{epoch:02d}-{val_loss:.2f}.hdf5'#参数什么意思？如何保存的name为ver2
ckpt = ModelCheckpoint(save_path)
# 网络训练常用fit或fit_genertaor，两者区别在于前者必须将数据全部读入内存，这在实际应用中很少能做到，所以常用后者，后者需要传入一个generator
# generator函数参照python的yield用法，另外，若不方便使用划分epoch，则可以用for循环和train_on_batch训练
# callbacks回调个人常用ModelCheckpoint和EarlyStopping或继承CallBack实现一些自己的功能
net.fit_generator(tg, len(td) // batchsize, epochs, callbacks=[ckpt], validation_data=vg,
                  validation_steps=len(vd) // batchsize)

