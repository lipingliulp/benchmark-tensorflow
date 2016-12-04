'''
Computation time comparison: tensor calculation v.s. for-loop with list of tensors 
Author: Li-Ping Liu 
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

size = 10000

# allocate two vectors, use different ways to calculate c = a + b 
a_np = np.random.rand(size)
b_np = np.random.rand(size)

a_tf = tf.constant(a_np)
b_tf = tf.constant(b_np)

a_tflist = tf.unpack(a_tf)
b_tflist = tf.unpack(b_tf)

a_nplist = a_np.tolist()
b_nplist = b_np.tolist()

# construct graph with list
t1_start =  time.clock()

c_tflist = list()
for i in xrange(size):
    c_tflist.append(a_tflist[i] + b_tflist[i])
c_tf1 = tf.pack(c_tflist)

t1_graph = time.clock() - t1_start
print("Construction of a graph with lists takes time %f seconds." % t1_graph)

# construct graph with vector 
t2_start =  time.clock()

c_tf2 = a_tf + b_tf

t2_graph = time.clock() - t2_start
print("Construction of a graph with tensors takes time %f seconds." % t2_graph)

session = tf.Session()

# calculation with graph constructed from lists 
t3_start =  time.clock()

c_np1 = session.run(c_tf1)

t3_calculation =  time.clock() - t3_start
print("Execution of a graph with lists takes time %f seconds." % t3_calculation)

# calculation with graph constructed from vectors 
t4_start =  time.clock()

c_np2 = session.run(c_tf2)

t4_calculation =  time.clock() - t4_start
print("Execution of a graph with tensors takes time %f seconds." % t4_calculation)

# you may think that python list operations take too much time
# time of numpy calculation with list 
t5_start = time.clock()
c_nplist = list()
for i in xrange(size):
    c_nplist.append(a_nplist[i] + b_nplist[i])
c_np0 = np.array(c_nplist)
t5_nplist = time.clock() - t5_start
print("Numpy calculation with lists takes time %f seconds." % t5_nplist)

assert(np.sum(c_np0 != c_np1) == 0)
assert(np.sum(c_np1 != c_np2) == 0)

