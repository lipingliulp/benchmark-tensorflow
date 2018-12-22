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


    values = tf.reshape(nz_values, [-1])
    for_fetch = tf.concat([tf.zeros([1], dtype=tf.float64), values], axis=0) 

    slice_indices = tf.tile(tf.reshape(tf.range(nslice), [nslice, 1, 1]), [1, nrow, ncol])
    row_indices = tf.tile(tf.reshape(tf.range(nrow), [1, nrow, 1]), [nslice, 1, ncol])
    
    indices = tf.stack([slice_indices, row_indices, col_indices], axis=3)
    indices = tf.reshape(indices, [-1, 3])

    fetch_ind = tf.Variable(tf.zeros([nslice, nrow, ncol_full], dtype=tf.int32))
    
    assign_zero = tf.assign(fetch_ind, tf.zeros(tf.shape(fetch_ind), dtype=tf.int32))
    with tf.control_dependencies([assign_zero]):
        fetch_ind = tf.scatter_nd_add(fetch_ind, indices, tf.range(nslice * nrow * ncol, dtype=tf.int32) + 1)

    dense = tf.gather(for_fetch, fetch_ind) 
    
    matmul = tf.matmul(dense, dense, transpose_b=True)

    return matmul 

if __name__ == '__main__':
    mod = tf.load_op_library('./ssmatmul_op.so')
    np.random.seed(9)

    nrow = 5 
    ncol = 8 
    ncol_full =  30  
    nslice =100 

    nz_values, col_indices, np_matmul = rand_problem(nrow, ncol, nslice, ncol_full)

    tf_values = tf.Variable(nz_values, dtype=tf.float64)
    tf_colind = tf.constant(col_indices, dtype=tf.int32)


    tf_matmul_1 = calculate_via_todense(tf_values, tf_colind, ncol_full)
    #tf_matmul_1 = tf.matmul(tf_values, tf_values, transpose_b=True)
    tf_matmul_2 = mod.ss_mat_mul(tf_values, tf_colind)


    tf_rand_mat = tf.constant(np.random.random_sample([nslice, nrow, nrow]), dtype=tf.float64)

    obj1 = tf.reduce_sum(tf.nn.sigmoid(tf.matmul(tf.sin(tf_matmul_1 * 2), tf_rand_mat)))
    obj2 = tf.reduce_sum(tf.nn.sigmoid(tf.matmul(tf.sin(tf_matmul_2 * 2), tf_rand_mat)))


    optimizer = tf.train.AdamOptimizer(1e-4)
    grad_var1 = optimizer.compute_gradients(obj1, [tf_values])
    grad_var2 = optimizer.compute_gradients(obj2, [tf_values])


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    matmul1 = sess.run(tf_matmul_1) 
    matmul2 = sess.run(tf_matmul_2) 

    gv1 = sess.run(grad_var1)
    gv2 = sess.run(grad_var2)


value_diff1 = np.sum(np.abs(matmul1 - np_matmul))
value_diff2 = np.sum(np.abs(matmul2 - np_matmul))



print('Value differences are %f and %f' % (value_diff1, value_diff2))

print('Gradient and value differences')

print(np.sum(np.abs(gv1[0][0] - gv2[0][0])))
print(np.sum(np.abs(gv1[0][1] - gv2[0][1])))




