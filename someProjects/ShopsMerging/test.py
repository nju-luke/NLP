# -*- coding: utf-8 -*-
# @Time    : 16-9-20 下午2:46
# @Author  : Luke
# @Software: PyCharm
import tensorflow as tf
import numpy as np

def add_vars():
    with tf.variable_scope("test"):
        tf.get_variable("test", shape=(3, 3))
        tf.get_variable("vec",shape=(3,))


with tf.variable_scope("test"):
    test = tf.get_variable("test", shape=(3, 3))
    tf.get_variable("vec", shape=(3,))

input_placeholder = tf.placeholder(tf.float32,shape=[3,5])
index_placeholder = tf.placeholder(tf.int32,None)

# vec = tf.slice(input_placeholder,[0,0],[2,5])
vec = tf.gather(input_placeholder,index_placeholder)
# vec = tf.squeeze(vec)
vec = tf.reduce_sum(vec,reduction_indices=0)

array = np.random.rand(3,5)
index = [[1,1]]
print array

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
# test,vecs,res = sess.run([test,vecs,res])
vecI = sess.run(vec,feed_dict={input_placeholder:array,index_placeholder:index})

print vecI

# print sess.run(tf.range(5,10))