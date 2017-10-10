'''
Will tensorflow reuse operations. suppose z1 and z2 are both defined as z1 = tf.add(a, b) and  z2 = tf.add(a, b), will z2 use the 
calculation result of z1? The answer is no.
Author: Li-Ping Liu 
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

# allocate an np array 
size = int(1e6)
niter1 = 1
niter2 = 100

npx = np.random.rand(size)

# calculate log(sigmoid(X)) in two different ways 
x = tf.constant(npx)


t0_start =  time.clock()

y1 = tf.identity(x)

t_graph = time.clock() - t0_start
print("Linking up %d identity_op takes time %f seconds." % (niter1, t_graph))


t0_start =  time.clock()

y2 = tf.identity(x)
for it in xrange(niter2 - 1):
    y2 = tf.identity(y2)

t_graph = time.clock() - t0_start
print("Linking up %d identity_ops takes time %f seconds." % (niter2, t_graph))

# computation
session = tf.Session()

# calculation with graph log(sigmoid(x))
t1_start =  time.clock()

npy1 = session.run(y1)

t_calculation =  time.clock() - t1_start
print("Calculation of %d identity_op takes time %f seconds." % (niter1, t_calculation))

# calculation with graph log(sigmoid(x))
t1_start =  time.clock()

npy2 = session.run(y2)

t_calculation =  time.clock() - t1_start
print("Calculation of %d identity_ops takes time %f seconds." % (niter2, t_calculation))



# conclusion: tensorflow will not merge nodes in computational graph
