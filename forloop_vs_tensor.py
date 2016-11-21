import tensorflow as tf
import numpy as np
import time


size = 1000
a = tf.constant(np.random.rand(size))
b = tf.constant(np.random.rand(size))
a_list = tf.unpack(a)
b_list = tf.unpack(b)

# construct graph with list
t1_start =  time.clock()

c_list = list()
for i in xrange(size):
    c_list.append(a_list[i] + b_list[i])
c1 = tf.pack(c_list)

t1_graph = time.clock() - t1_start
print("Construction of a graph with lists takes time %f seconds." % t1_graph)

# construct graph with vector 
t2_start =  time.clock()
c2 = a + b
t2_graph = time.clock() - t2_start
print("Construction of a graph with tensors takes time %f seconds." % t2_graph)

session = tf.Session()

# calculation with graph constructed from lists 
t3_start =  time.clock()
npc1 = session.run(c1)
t3_calculation =  time.clock() - t3_start
print("Execution of a graph with lists takes time %f seconds." % t3_calculation)

# calculation with graph constructed from vectors 
t4_start =  time.clock()
npc2 = session.run(c2)
t4_calculation =  time.clock() - t4_start
print("Execution of a graph with tensors takes time %f seconds." % t4_calculation)

assert(np.sum(npc1 != npc2) == 0)

