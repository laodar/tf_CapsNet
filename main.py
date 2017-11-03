import tensorflow as tf
from CapsNet import *
import cv2
import numpy as np
from data import *

batch_size = 32
is_multi_mnist = False
is_shift_ag = True
irun = 0
num_show = 5
lr = 0.001
steps = 100000
save_frequence = 10000
decay_frequence = 5000
is_show_multi_rec = True
is_show_sample = True
key = -1
min_lr = 5e-6

if is_multi_mnist:
    train_iter = multimnist_train_iter(iters=steps,batch_size=batch_size,is_shift_ag = True)
    test_iter = multimnist_test_iter(iters=steps, batch_size=batch_size,is_shift_ag = True)
else:
    train_iter = mnist_train_iter(iters=steps,batch_size=batch_size,is_shift_ag = True)
    test_iter = mnist_test_iter(iters=steps, batch_size=batch_size,is_shift_ag = True)
multi_iter = multimnist_test_iter(iters=steps,batch_size=num_show,is_shift_ag = True)

net = CapsNet(is_multi_mnist=is_multi_mnist)
tf.summary.scalar('error_rate_on_test_set', (1.0 - net.accuracy) * 100.0)
tf.summary.scalar('loss_reconstruction_on_test_set', net.loss_rec)
merged = tf.summary.merge_all()
init = tf.initialize_all_variables()

sess = tf.Session()
writer = tf.summary.FileWriter("./sum",sess.graph)
saver = tf.train.Saver()

sess.run(init)

for X,Y in train_iter:
    X_TEST, Y_TEST = test_iter.next()

    LS, LS_REC, ACC, _ = sess.run([net.loss, net.loss_rec, net.accuracy, net.train], feed_dict={net.x: X, net.y: Y, net.lr: lr})
    ACC_TEST, result = sess.run([net.accuracy,merged], feed_dict={net.x: X_TEST, net.y: Y_TEST})

    writer.add_summary(result, irun)

    print irun, LS, LS_REC, ACC, ACC_TEST

    if is_show_sample:
        H_SAM = np.random.randn(num_show * 10, 10, 16)
        H_SAM = H_SAM / (0.001 + np.sum(H_SAM ** 2.0, axis=-1, keepdims=True) ** 0.5)
        Y_SAM = np.eye(10)[np.array(range(10) * num_show)].astype(float)
        X_SAM = sess.run(net.x_sample, feed_dict={net.h_sample: H_SAM, net.y_sample: Y_SAM})
        images_sample = X_SAM.reshape([num_show, 10, 28, 28, 1])
        images_sample = np.concatenate(images_sample, axis=1)
        images_sample = cv2.resize(np.concatenate(images_sample, axis=1), dsize=(0, 0), fx=3, fy=3)
        cv2.imshow('SampleFromH', images_sample)

    if is_show_multi_rec:
        X_MULTI,Y_MULTI = multi_iter.next()
        X_REC1,X_REC2 = sess.run(net.x_recs, feed_dict={net.x: X_MULTI, net.y: Y_MULTI})
        # turn the composed image to be 3 channel gray image
        images_org = np.stack([X_MULTI[:num_show,:,:,0]]*3,axis=-1)
        black = np.zeros([num_show, 28, 28, 1])
        images_recs = np.concatenate([black, X_REC1, X_REC2], axis=-1)
        images_rec1 = np.concatenate([black, black, X_REC2], axis=-1)
        images_rec2 = np.concatenate([black, X_REC1, black], axis=-1)
        image_show = np.concatenate([images_org, images_recs, images_rec1, images_rec2], axis=2)
        image_show = cv2.resize(np.concatenate(image_show, axis=0), dsize=(0, 0), fx=3, fy=3)
        cv2.imshow('MultiMnistReconstruction', image_show)

    if is_show_multi_rec or is_multi_mnist:
        key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite('MultiMnistReconstruction%d.png' % irun, image_show * 255.0)
        cv2.imwrite('SampleFromH%d.png' % irun, images_sample * 255.0)

    if irun+1 % save_frequence == 0:
        saver.restore(sess, tf.train.get_checkpoint_state('./cpt/').model_checkpoint_path)

    if irun+1 % decay_frequence == 0 and lr > min_lr:
        lr *= 0.5

    irun += 1