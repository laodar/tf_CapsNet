import tensorflow as tf
from CapsNet import *
import cv2
import numpy as np
from data import *

batch_size = 64
is_multi_mnist = False
irun = 0
num_show = 5

net = CapsNet(is_multi_mnist=is_multi_mnist)
tf.summary.scalar('error_rate_on_test_set', (1.0 - net.accuracy) * 100.0)
tf.summary.scalar('loss_reconstruction_on_test_set', net.loss_rec)
merged = tf.summary.merge_all()
init = tf.initialize_all_variables()

sess = tf.Session()

writer = tf.summary.FileWriter("./sum",sess.graph)

sess.run(init)

if is_multi_mnist:
    train_iter = multimnist_train_iter(iters=100000,batch_size=batch_size)
    test_iter = multimnist_test_iter(iters=100000, batch_size=batch_size)
else:
    train_iter = mnist_train_iter(iters=100000,batch_size=batch_size)
    test_iter = mnist_test_iter(iters=100000, batch_size=batch_size)

for X,Y in train_iter:
    X_TEST, Y_TEST = test_iter.next()
    H_SAM = np.random.randn(num_show * 10, 10, 16)
    H_SAM = H_SAM / (0.001 + np.sum(H_SAM ** 2.0, axis=-1, keepdims=True) ** 0.5)
    Y_SAM = np.eye(10)[np.array(range(10) * num_show)].astype(float)

    LS, LS_REC, ACC, _ = sess.run([net.loss, net.loss_rec, net.accuracy, net.train], feed_dict={net.x: X, net.y: Y})
    ACC_TEST, result = sess.run([net.accuracy,merged], feed_dict={net.x: X_TEST, net.y: Y_TEST})

    writer.add_summary(result, irun)

    print irun, LS, LS_REC, ACC, ACC_TEST

    X_SAM = sess.run(net.x_sample, feed_dict={net.h_sample: H_SAM, net.y_sample: Y_SAM})

    if is_multi_mnist:
        X_MULTI = X_TEST[:num_show]
        Y_MULTI = Y_TEST[:num_show]
        X_REC1,X_REC2 = sess.run(net.x_recs, feed_dict={net.x: X_MULTI, net.y: Y_MULTI})
    else:
        X_MULTI = np.logical_or(X[:num_show], X_TEST[:num_show]).astype(np.float32)
        X_MULTI = np.concatenate([X_MULTI, X_MULTI], axis=0)
        Y_MULTI = np.concatenate([Y[:num_show], Y_TEST[:num_show]], axis=0)
        X_RECs = sess.run(net.x_recs, feed_dict={net.x: X_MULTI, net.y: Y_MULTI})[0]
        X_REC1, X_REC2 = X_RECs[num_show:],X_RECs[:num_show]
    images_org = np.concatenate([X_MULTI[:num_show]] * 3, axis=-1)  # Turn it to 3 channel
    black = np.zeros([num_show, 28, 28, 1])
    images_recs = np.concatenate([black, X_REC1, X_REC2], axis=-1)
    images_rec1 = np.concatenate([black, black, X_REC2], axis=-1)
    images_rec2 = np.concatenate([black, X_REC1, black], axis=-1)
    image_show = np.concatenate([images_org, images_recs, images_rec1, images_rec2], axis=2)
    image_show = cv2.resize(np.concatenate(image_show, axis=0), dsize=(0, 0), fx=3, fy=3)
    images_sample = X_SAM.reshape([num_show, 10, 28, 28, 1])
    images_sample = np.concatenate(images_sample, axis=1)
    images_sample = cv2.resize(np.concatenate(images_sample, axis=1), dsize=(0, 0), fx=3, fy=3)

    cv2.imshow('MultiMnistReconstruction', image_show)
    cv2.imshow('SampleFromH', images_sample)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite('MultiMnistReconstruction%d.png' % irun, image_show * 255.0)
        cv2.imwrite('SampleFromH%d.png' % irun, images_sample * 255.0)

    irun += 1