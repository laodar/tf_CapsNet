from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np

def augmentation(x,max_offset=2):
    bz,h,w,c = x.shape
    bg = np.zeros([bz,w+2*max_offset,h+2*max_offset,c])
    offsets = np.random.randint(0,2*max_offset+1,2)
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w,:] = x
    return bg[:,max_offset:max_offset+h,max_offset:max_offset+w,:]

def mnist_train_iter(iters=1000,batch_size=32,is_shift_ag=True):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    max_offset = int(is_shift_ag)*2
    for i in range(iters):
        batch = mnist.train.next_batch(batch_size)
        images = batch[0].reshape([batch_size, 28, 28, 1])
        images = np.concatenate([images]*3,axis=-1)
        yield augmentation(images,max_offset), np.stack([batch[1]]*3, axis=-1)

def mnist_test_iter(iters=1000,batch_size=32,is_shift_ag=False):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    max_offset = int(is_shift_ag) * 2
    for i in range(iters):
        batch = mnist.test.next_batch(batch_size)
        images = batch[0].reshape([batch_size, 28, 28, 1])
        images = np.concatenate([images] * 3, axis=-1)
        yield augmentation(images,max_offset), np.stack([batch[1]]*3, axis=-1)

def multimnist_train_iter(iters=1000,batch_size=32,is_shift_ag=True):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    max_offset = int(is_shift_ag) * 2
    for i in range(iters):
        batch1 = mnist.train.next_batch(batch_size)
        batch2 = mnist.train.next_batch(batch_size)
        images1 = augmentation(batch1[0].reshape([batch_size, 28, 28, 1]),max_offset)
        images2 = augmentation(batch2[0].reshape([batch_size, 28, 28, 1]),max_offset)
        images = np.logical_or(images1,images2).astype(np.float32)
        images = np.concatenate([images,images1,images2], axis=-1)
        y1,y2 = batch1[1],batch2[1]
        y0 = np.logical_or(y1,y2).astype(np.float32)
        yield images, np.stack([y0,y1,y2], axis=-1)

def multimnist_test_iter(iters=1000,batch_size=32,is_shift_ag=True):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    max_offset = int(is_shift_ag) * 2
    for i in range(iters):
        batch1 = mnist.test.next_batch(batch_size)
        batch2 = mnist.test.next_batch(batch_size)
        images1 = augmentation(batch1[0].reshape([batch_size, 28, 28, 1]),max_offset)
        images2 = augmentation(batch2[0].reshape([batch_size, 28, 28, 1]),max_offset)
        images = np.logical_or(images1,images2).astype(np.float32)
        images = np.concatenate([images,images1,images2], axis=-1)
        y1,y2 = batch1[1],batch2[1]
        y0 = np.logical_or(y1,y2).astype(np.float32)
        yield images, np.stack([y0,y1,y2], axis=-1)

'''
m_train_iter = mnist_train_iter()
m_test_iter = mnist_test_iter()
mm_train_iter = multimnist_train_iter()
mm_test_iter = multimnist_test_iter()
for i in range(10):
    img_mtrain = m_train_iter.next()[0][0]
    img_mtest = m_test_iter.next()[0][0]
    img_mmtrain = mm_train_iter.next()[0][0]
    img_mmtest = mm_test_iter.next()[0][0]
    cv2.imshow('mnist_train',cv2.resize(img_mtrain,dsize=(128,128)))
    cv2.imshow('mnist_test', cv2.resize(img_mtest,dsize=(128,128)))
    cv2.imshow('multimnist_train', cv2.resize(img_mmtrain,dsize=(128,128)))
    cv2.imshow('multimnist_test', cv2.resize(img_mmtest,dsize=(128,128)))
    cv2.waitKey(-1)
'''