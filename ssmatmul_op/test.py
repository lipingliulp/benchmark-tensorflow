import numpy as np
import tensorflow as tf
import scipy.sparse
import _ssmatmul_grad

def rand_problem(nrow, ncol, nslice, ncol_full):

    nz_values = []
    col_indices = []
   
    matmul = []

    for islice in range(nslice):

        nz_slice = np.random.random_sample([nrow, ncol])
        col_slice = np.zeros([nrow, ncol], dtype=np.int32)
        for i in range(nrow):
            col_slice[i, :] = np.random.choice(np.arange(ncol_full), size=ncol, replace=False)
        col_slice.sort(axis=1)

        nz_values.append(nz_slice)
        col_indices.append(col_slice)


        row_slice = np.outer(np.arange(nrow), np.ones(ncol))
        dense = scipy.sparse.coo_matrix((nz_slice.flatten(), (row_slice.flatten(), col_slice.flatten())), shape=(nrow, ncol_full)).todense()
        matmul_slice = np.array(dense.dot(dense.T))
        matmul.append(matmul_slice)



    nz_values = np.stack(nz_values)
    col_indices = np.stack(col_indices)
    matmul = np.stack(matmul)
    
    return nz_values, col_indices, matmul



def calculate_via_todense(nz_values, col_indices, ncol_full):

    nslice = nz_values.get_shape()[0]
    nrow = nz_values.get_shape()[1]
    ncol = nz_values.get_shape()[2]

    slice_range = tf.reshape(tf.range(nslice), [nslice, 1, 1])
    col_indices = col_indices + slice_range * ncol_full

    row_indices = tf.tile(tf.reshape(tf.range(nrow), [1, nrow, 1]), [1, 1, ncol])
    row_indices = row_indices + slice_range * nrow
    
    
    values = tf.reshape(nz_values, [-1])
    for_fetch = tf.stack([values, tf.zeros([1])]) 

    indices = tf.stack([tf.reshape(row_indices, [-1]), tf.reshape(col_indices, [-1])], axis=1) 
    
    full_shape = [nslice * nrow, nslice * ncol_full] 
    dense = tf.sparse_to_dense(indices, full_shape, values)

    
    matmul = tf.matmul(dense, tf.transpose(dense))

    matmul = tf.transpose(tf.reshape(matmul, [nslice, nrow, nslice, nrow]), [0, 2, 1, 3])
    

    diag_ind = tf.stack([tf.range(nslice), tf.range(nslice)], axis=1)

    matmul = tf.gather_nd(matmul, diag_ind)

    return matmul 

if __name__ == '__main__':
    mod = tf.load_op_library('./ssmatmul_op.so')
    np.random.seed(9)

    nrow = 10 
    ncol = 8 
    ncol_full = 100 
    nslice = 5 

    nz_values, col_indices, np_matmul = rand_problem(nrow, ncol, nslice, ncol_full)

    tf_values = tf.Variable(nz_values, dtype=tf.float64)
    tf_colind = tf.constant(col_indices, dtype=tf.int32)


    tf_matmul_1 = calculate_via_todense(tf_values, tf_colind, ncol_full)
    tf_matmul_2 = mod.ss_mat_mul(tf_values, tf_colind)


    tf_rand_mat = tf.random_uniform(tf_matmul_2.shape, dtype=tf.float64)

    obj1 = tf.reduce_sum(tf.matmul(tf_matmul_1, tf_rand_mat, transpose_b=True))
    obj2 = tf.reduce_sum(tf.matmul(tf_matmul_2, tf_rand_mat, transpose_b=True))


    optimizer = tf.train.AdamOptimizer(1e-4)
    #grad_var1 = optimizer.compute_gradients(obj1, [tf_values])
    grad_var2 = optimizer.compute_gradients(obj2, [tf_values])


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    matmul1 = sess.run(tf_matmul_1) 
    matmul2 = sess.run(tf_matmul_2) 

    #gv1 = sess.run(grad_var1)
    gv2 = sess.run(grad_var2)


value_diff1 = np.sum(np.abs(matmul1 - np_matmul))
value_diff2 = np.sum(np.abs(matmul2 - np_matmul))

print('Value differences are %f and %f' % (value_diff1, value_diff2))




