'''
Will tensorflow merge two calculation nodes into one?
Example: tf.log(tf.sigmoid(x)) = - tf.nn.softplus(-x)
Author: Li-Ping Liu 
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

# allocate an np array 
size = int(1e5)
scale = 10 # if scale is large enough, log(sigmoid(X)) will fail
npX = (np.random.rand(size) - 0.5) * scale 

# calculate log(sigmoid(X)) in two different ways 
X = tf.constant(npX)
Y1 = tf.log(tf.sigmoid(X))
Y2 = - tf.nn.softplus(- X)


# computation
session = tf.Session()

# calculation with graph log(sigmoid(x))
t1_start =  time.clock()

npY1 = session.run(Y1)

t1_calculation =  time.clock() - t1_start
print("Calculation of log(sigmoid(X)) takes time %f seconds." % t1_calculation)


# calculation with graph - softplus(- x)
t2_start =  time.clock()

npY2 = session.run(Y1)

t2_calculation =  time.clock() - t2_start
print("Calculation of - softplus(-X) takes time %f seconds." % t2_calculation)

print(npY1.shape)
print(np.argmax(npY1 - npY2))

print(np.mean(np.abs(npY1 - npY2)))

# conclusion: tensorflow will not merge nodes in computational graph
