import tensorflow as tf
from ops import *
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np

def mnist_train_iter(iters=1000,batch_size=32):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    for i in range(iters):
        batch = mnist.train.next_batch(batch_size)
        images = batch[0].reshape([batch_size,28,28,1])
        yield images,batch[1]

def mnist_test_iter(iters=1000,batch_size=32):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    for i in range(iters):
        batch = mnist.test.next_batch(batch_size)
        images = batch[0].reshape([batch_size,28,28,1])
        yield images,batch[1]

def squash(s,axis=-1):
    length_s = tf.reduce_sum(s ** 2.0, axis=axis,keep_dims=True) ** 0.5
    v = s*length_s/(1.0+length_s**2.0)
    return v

def get_CapsNet(x,iterations = 3,reuse = False):
    with tf.variable_scope('CapsNet',reuse=reuse):
        wconv1 = tf.get_variable('wconv1',[9,9,1,256],initializer=tf.truncated_normal_initializer(stddev=0.02))
        bconv1 = tf.get_variable('bconv1', [256], initializer=tf.truncated_normal_initializer(stddev=0.02))
        wconv2 = tf.get_variable('wconv2',[9,9,256,8*32],initializer=tf.truncated_normal_initializer(stddev=0.02))
        bconv2 = tf.get_variable('bconv2', [8*32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        #[batch,i_row,i_column,i_channel,u,j,v]
        #[0    ,1    ,2       ,3        ,4,5,6]
        wcap = tf.get_variable('wcap',[1,6,6,32,8,10,16],initializer=tf.truncated_normal_initializer(stddev=0.2))
        b = tf.get_variable('coupling_coefficient_logits',[1,6,6,32,1,10,1],initializer=tf.constant_initializer(0.0))

    c = tf.stop_gradient(tf.nn.softmax(b, dim=5))

    conv1 = relu(conv1x1(x,wconv1)+bconv1)
    s_primary = conv2x2(conv1,wconv2)+bconv2 #with shape [batch_size,6,6,8*32]
    s_primary = tf.reshape(s_primary,[-1,6,6,32,8,1,1])
    v_primary = squash(s_primary,axis=4)

    u = v_primary
    u_ = tf.reduce_sum(u*wcap,axis=[4],keep_dims=True)
    s = tf.reduce_sum(u_*c,axis=[1,2,3],keep_dims=True)
    v = squash(s,axis=-1)

    #u_ with shape [batch_size,6,6,32,1,10,16]
    #v  with shape [batch_size,1,1, 1,1,10,16]

    for i in range(iterations-1):
        b += tf.reduce_sum(u_*v,axis=-1,keep_dims=True)
        c =  tf.nn.softmax(b, dim=5)
        s = tf.reduce_sum(u_ * c, axis=[1, 2, 3], keep_dims=True)
        v = squash(s,axis=-1)

    v_digit = tf.squeeze(v) #v_digit with shape [batch_size,10,16]

    return v_digit,c

def get_mlp_decoder(h,num_h=[10*16,512,1024,784],reuse=False):
    h = tf.reshape(h,[-1,10*16])
    with tf.variable_scope('decoder',reuse=reuse):
        weights = []
        for i in range(len(num_h)-1):
            w = tf.get_variable('wfc%d'%i,[num_h[i],num_h[i+1]],initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('bfc%d'%i,[num_h[i+1]],initializer=tf.truncated_normal_initializer(stddev=0.02))
            weights.append((w,b))
            if i<len(num_h)-2:
                h = relu(fullyconnect(h,w,b))
            else:
                h = tf.nn.sigmoid(fullyconnect(h,w,b))
    x_rec = tf.reshape(h,[-1,28,28,1])
    return x_rec#,weights

is_multi_mnist = 1.0
x = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,10])
h_sample = tf.placeholder(tf.float32,[None,10,16])
y_sample = tf.placeholder(tf.float32,[None,10])

v,c = get_CapsNet(x,iterations=3)

x_rec = get_mlp_decoder(v*y[:,:,None])

x_sample = get_mlp_decoder(h_sample*y_sample[:,:,None],reuse=True)

length_v = tf.reduce_sum(v**2.0,axis=-1)**0.5 #length_v with shape [batch_size,10]

loss_cls = tf.reduce_sum(y*tf.maximum(0.0,0.9-length_v)**2.0+0.5*(1.0-y)*tf.maximum(0.0,length_v-0.1)**2.0,axis=-1)

loss_rec = tf.reduce_sum((x_rec-x)**2.0,axis=[1,2,3]))

enable_mask = is_multi_mnist*(tf.reduce_sum(y,axis=-1,keep_dims=True) - 1.0) + (1.0-is_multi_mnist)*1.0

loss = tf.reduce_sum((loss_cls + 0.0005*loss_rec)*enable_mask)/tf.reduce_sum(enable_mask)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(length_v,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train = tf.train.AdamOptimizer().minimize(loss)

init = tf.initialize_all_variables()

tf.summary.scalar('error_rate_on_test_set',(1.0-accuracy)*100.0)

tf.summary.scalar('loss_reconstruction_on_test_set',loss_rec)

merged = tf.summary.merge_all()

sess = tf.Session()

writer = tf.summary.FileWriter("./sum",sess.graph)

sess.run(init)

test_iter = mnist_test_iter(iters=100000,batch_size=128)

irun = 0

num_show = 5

for X,Y in mnist_train_iter(iters=100000,batch_size=128):

    X_TEST, Y_TEST = test_iter.next()
    X_MULTI = np.logical_or(X[:num_show],X_TEST[:num_show]).astype(np.float32)
    X_MULTI = np.concatenate([X_MULTI,X_MULTI],axis=0)
    Y_MULTI = np.concatenate([Y[:num_show],Y_TEST[:num_show]],axis=0)
    H_SAM = np.random.rand(num_show*10,10,16)
    H_SAM = H_SAM/(0.0001+np.sum(H_SAM**2.0,axis=-1,keepdims=True)**0.5)
    Y_SAM = np.eye(10)[np.array(range(10)*num_show)].astype(float)

    LS,LS_REC,ACC,_ = sess.run([loss,loss_rec,accuracy,train],feed_dict={x:X,y:Y})
    ACC_TEST,result = sess.run([accuracy,merged], feed_dict={x: X_TEST, y: Y_TEST})
    X_REC = sess.run(x_rec,feed_dict={x:X_MULTI,y:Y_MULTI})
    X_SAM = sess.run(x_sample,feed_dict={h_sample:H_SAM,y_sample:Y_SAM})

    writer.add_summary(result, irun)

    print irun, LS, LS_REC, ACC, ACC_TEST

    images_org = np.concatenate([X_MULTI[:num_show]]*3,axis=-1) #Turn it to 3 channel
    black = np.zeros([num_show,28,28,1])
    images_recs = np.concatenate([black,X_REC[num_show:],X_REC[:num_show]],axis=-1)
    images_rec1 = np.concatenate([black,black,X_REC[:num_show]], axis=-1)
    images_rec2 = np.concatenate([black, X_REC[num_show:], black], axis=-1)
    image_show = np.concatenate([images_org,images_recs,images_rec1,images_rec2],axis=2)
    image_show = cv2.resize(np.concatenate(image_show, axis=0),dsize=(0,0),fx=3,fy=3)
    images_sample =  X_SAM.reshape([num_show,10,28,28,1])
    images_sample = np.concatenate(images_sample,axis=1)
    images_sample = cv2.resize(np.concatenate(images_sample,axis=1),dsize=(0,0),fx=3,fy=3)

    cv2.imshow('MultiMnistReconstruction', image_show)
    cv2.imshow('SampleFromH',images_sample)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite('MultiMnistReconstruction%d.png'%irun,image_show*255.0)
        cv2.imwrite('SampleFromH%d.png'%irun, images_sample*255.0)

    irun += 1


