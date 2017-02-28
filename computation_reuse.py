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

npx = np.random.rand(size)

# calculate log(sigmoid(X)) in two different ways 
x = tf.constant(npx)

y1 = tf.reduce_mean(x)
y2 = tf.reduce_mean(x)

z1 = y1 + y1 
z2 = y1 + y2 

# computation
session = tf.Session()

# calculation with graph log(sigmoid(x))
t1_start =  time.clock()

npz1 = session.run(z1)

t1_calculation =  time.clock() - t1_start
print("Calculation of z1 takes time %f seconds." % t1_calculation)


# calculation with graph - softplus(- x)
t2_start =  time.clock()

npz2 = session.run(z2)

t2_calculation =  time.clock() - t2_start
print("Calculation of z2 takes time %f seconds." % t2_calculation)


print('difference of two results is ', np.abs(npz1 - npz2))

# conclusion: tensorflow will not merge nodes in computational graph
