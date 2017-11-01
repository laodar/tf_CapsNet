import tensorflow as tf

def fullyconnect(h,w,b):
    return tf.add(tf.matmul(h,w),b)

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, leaky=0.1):
    return tf.maximum(x, leaky * x)

def conv2x2(x, w,padding='VALID'):
    return tf.nn.conv2d(x, w, [1, 2, 2, 1], padding=padding)

def conv1x1(x, w,padding='VALID'):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding=padding)