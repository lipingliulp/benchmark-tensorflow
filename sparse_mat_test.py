import tensorflow as tf
import numpy as np
import time


def vec_mul_sparse_mat(a, row_ind, col_ind, b):

    m = a * tf.gather(b, col_ind)
    a_cum = tf.cumsum(m) 
    
    #ri_pad = tf.pad(row_ind - (row_ind[-1] + 1), [[0, 1]]) + (row_ind[-1] + 1) 
    flag = (row_ind[1:] - row_ind[:-1]) > 0
    flag = tf.logical_not(tf.pad(tf.logical_not(flag), [[0, 1]]))

    s_cum = tf.boolean_mask(a_cum, flag)

    s_pad = tf.pad(s_cum, [[1, 0]])

    s = s_pad[1:] - s_pad[:-1]

    return s

def sparse_mat(a, row_ind, col_ind, b, shape):


    ind = tf.concat([tf.expand_dims(row_ind, axis=1), tf.expand_dims(col_ind, axis=1)], axis=1)
    A = tf.SparseTensor(indices=ind, values=a, dense_shape=shape)
    tfb = tf.constant(b)
    b = tf.expand_dims(tfb, axis=1)
    s = tf.sparse_tensor_dense_matmul(A, b)

    return s



# 1000 non-zero elements
N = 1000
nrow = 100
ncol = 100
a = np.random.rand(N)
ind = np.random.randint(low=1, high=10, size=N)
ind = np.cumsum(ind)
row_ind = ind / ncol 


col_ind = ind % ncol 
shape=[np.max(row_ind) + 1, np.max(col_ind) + 1]

# remove elements in rows after first nrow rows
#a = row_ind[row_ind < nrow]
#row_ind = row_ind[row_ind < nrow]
#col_ind = col_ind[row_ind < nrow]

b = np.random.rand(ncol)


tfc1 = sparse_mat(a, row_ind, col_ind, b, shape)
tfc2 = vec_mul_sparse_mat(a, row_ind, col_ind, b)

session = tf.Session()


t1_start =  time.clock()
npc1 = session.run(tf.squeeze(tfc1))
t1_graph = time.clock() - t1_start
print('Time for first computation is %.3f' % t1_graph)


t2_start =  time.clock()
npc2 = session.run(tf.squeeze(tfc2))
t2_graph = time.clock() - t2_start

print('Time for second computation is %.3f ' % t2_graph)




diff = np.mean(np.abs(npc1 - npc2))

print('The difference is %f ' % diff)

print('Tensorflow version is ' + tf.__version__)

# The output is 
'''

'''
#Time: 15:15pm, 4/28/2017



