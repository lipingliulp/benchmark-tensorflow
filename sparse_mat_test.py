import tensorflow as tf
import numpy as np
import time


def cumsum_sparse_mat_mul(a, row_ind, col_ind, b):

    m = a * tf.gather(b, col_ind)
    a_cum = tf.cumsum(m) 
    
    #ri_pad = tf.pad(row_ind - (row_ind[-1] + 1), [[0, 1]]) + (row_ind[-1] + 1) 
    flag = (row_ind[1:] - row_ind[:-1]) > 0
    flag = tf.logical_not(tf.pad(tf.logical_not(flag), [[0, 1]]))

    s_cum = tf.boolean_mask(a_cum, flag)

    s_pad = tf.pad(s_cum, [[1, 0]])

    s = s_pad[1:] - s_pad[:-1]

    s = tf.expand_dims(s, axis=1)

    return s

def sparse_mat_mul(a, row_ind, col_ind, b, shape):

    ind = tf.concat([tf.expand_dims(row_ind, axis=1), tf.expand_dims(col_ind, axis=1)], axis=1)
    A = tf.SparseTensor(indices=ind, values=a, dense_shape=shape)
    tfb = tf.constant(b)
    b = tf.expand_dims(tfb, axis=1)
    s = tf.sparse_tensor_dense_matmul(A, b)


    return s



# 1000 non-zero elements
N = 1000
ncol = 100

# non-zero entries
a = np.random.rand(N).astype(np.float32)

# indices
ind = np.random.randint(low=1, high=10, size=N).astype(np.int32)
ind = np.cumsum(ind)
row_ind = ind // ncol 
col_ind = ind % ncol 

# dense_shape
shape=[np.max(row_ind) + 1, np.max(col_ind) + 1]

# a dense vector
b = np.random.rand(ncol).astype(np.float32)

cum_result = cumsum_sparse_mat_mul(a, row_ind, col_ind, b)
sparse_result = sparse_mat_mul(a, row_ind, col_ind, b, shape)

session = tf.Session()



t1_start =  time.clock()
cum_result_np = session.run(cum_result)
t1_graph = time.clock() - t1_start
print('Calculation with cumsum takes time %.3f seconds.' % t1_graph)



t2_start =  time.clock()
sparse_result_np = session.run(sparse_result)
t2_graph = time.clock() - t2_start
print('Sparse matrix multiplication takes time  %.3f seconds' % t2_graph)


diff = np.mean(np.abs(cum_result_np - sparse_result_np))

print('The difference is %f ' % diff)

print('Tensorflow version is ' + tf.__version__)

# The output is 
'''
The difference is 0.000011 
Tensorflow version is 1.4.1
'''
#Time: 15:15pm, 4/28/2017



