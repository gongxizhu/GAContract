import tensorflow as tf

'''
leaky relu: y = x if x>=0, y = ax if x<0 and 0=<a<1,
'''
def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)