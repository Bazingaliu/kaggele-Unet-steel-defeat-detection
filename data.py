import cv2
import os
import numpy as np
import pandas as pd


def read_data(path):
    data = pd.read_csv(path)
    img_ids, img_masks = data['ImageId_ClassId'], data['EncodedPixels']
    idx = list(range(0, len(img_ids), 4))
    img_ids = img_ids[idx]#arry直接传入一个list可以直接取出这个List位置下的元素。
    img_ids = np.array([s[:-2] for s in img_ids])#获得每一张图片的名字
    img_masks = np.array(img_masks)#取得所有的mask，大小为img_ids的四倍，转化为array
    img_masks = np.reshape(img_masks, [-1, 4])#转成二维的，参数[-1,4]表示第二个维度为4，第一个维度用总的/4
    # print(img_masks)
    return img_ids, img_masks


def split_nan(id, mask):
    nor_id, nan_id, nor_mask, nan_mask = [], [], [], []
    for i in range(len(mask)):#mask 是一个二维的，len（）取第一个维度
        zm = 0
        for m in mask[i]:
            if isinstance(m, float):
                zm += 1
        if zm == 4:
            nan_id.append(id[i])
            nan_mask.append(mask[i])# 一张图假如四中类型的缺陷不存在的话就存在nan_maks
        else:
            nor_id.append(id[i])
            nor_mask.append(mask[i]) #nor_mask用于储存那些一张图上大于或者等于1缺陷的
    nor_id = np.array(nor_id)
    nan_id = np.array(nan_id)
    nor_mask = np.array(nor_mask)
    nan_mask = np.array(nan_mask)
    return nor_id, nan_id, nor_mask, nan_mask


def split_all(id,mask):
    data_all, data_mask = [], []
    for i in range(len(mask)):
        data_mask.append(mask[i])
        data_all.append(id[i])
    data_mask = np.array(data_mask)
    data_all = np.array(data_all)
    return data_all, data_mask



def generator(img, mask, batchsize=16):
    batchnum = len(img) // batchsize  #// 表示整数除法，在batch_size为16的时候有多少个batch.
    idx = list(range(len(img)))#图片个数同size的List[0,1,2,...,size-1]
    while True:
        np.random.shuffle(idx) #对Train的数据顺序进行打乱
        for i in range(batchnum): 
            l = i * batchsize
            r = l + batchsize
            batchimg, batchmask = img[idx[l:r]], mask[idx[l:r]] #按照给出的batch_size取一个batch的图片
            batchimg = read_img(batchimg)#读取这个patch的照片
            batchmask = read_mask(batchmask)#读取对应的mask
            yield batchimg, batchmask #递归的输出值，每一次都会返回得到的值，用于在for,while中输出每一次迭代的值


def read_img(paths):#这个path是个array，表示一个batch的照片名
    imgs = []
    for s in paths:
        img = cv2.imread(os.path.join('../severstal-steel-defect-detection/train_images', s), cv2.IMREAD_GRAYSCALE)#以灰度的方式读取，不包括a通道
        img = cv2.resize(img, (512, 256))
        imgs.append(img) 
    imgs = np.array(imgs)# shape=(b,h,w,c)？？
    imgs = imgs / 255
    imgs = np.expand_dims(imgs, -1)#在最后的一个维度加入一个维度？？
    return imgs


def read_mask(masks):
    ms = []
    for m4 in masks:#m4表示每一行，表示一个照片的Mask个数，4个数据，可能为空
        mm = []
        for m in m4: #m是这一行的的数据，表示四个class
            if isinstance(m, float):#没有缺陷
                tmp = np.zeros((256, 512, 1))#生成一个whc直接为训练的数据大小，值为0的tensor，表示没有缺陷
            else:
                # tmp = pix2mask(m)
                tmp = rle2mask(m) #存在缺陷。返回一个mask
            # print(tmp.shape)
            mm.append(tmp)#储存4个class的Mask
        ms.append(np.concatenate(mm, -1))#在最后一个维度上组合成一个tensor
    ms = np.array(ms)#一张img此时就只对应一个ms了
    return ms


def rle2mask(mask_rle, shape=(1600, 256)):
    s = mask_rle.split()
    #np.asarray()和np.array()一样转化为np能处理的数据，但是除非必要，是不会cope对象的而np.array()是会cope对象的
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    #上式表示取mask中起点和每一个起点的步长
    starts -= 1
    ends = starts + lengths#对每一个start都存在一个end
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends): #zip函数打包合成数据，一个start对应一个end
        img[lo:hi] = 1#该处的mask值为1
    img = img.reshape(shape).T#旋转shape变为(1600,512)
    img = cv2.resize(img, (512, 256))#resize()这一步默认采用双线性插值，会改变灰度值
    img = np.where(img >= 0.5, np.ones_like(img), np.zeros_like(img))#将上一步的操作的灰度值从新变成0,1
    img = np.reshape(img, (256, 512, 1))
    return img


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()#旋转拉直 
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def pix2mask(pix):
    mask = np.zeros(256 * 1600)
    pix = np.reshape(pix.split(), [-1, 2])
    pix = pix.astype(np.int32)
    for s, l in pix:
        s = s - 1
        mask[s:(s + l)] = 255
    mask = np.reshape(mask, (256, 1600))
    mask = cv2.resize(mask, (512, 256))
    mask = np.reshape(mask, (256, 512, 1))
    mask[np.where(mask >= 127.5)] = 1
    mask[np.where(mask < 127.5)] = 0
    return mask




