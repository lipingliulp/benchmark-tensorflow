import tensorflow as tf

arr = tf.range(0, 1000, dtype=tf.float32) 

result1 = tf.sqrt(arr) 


alist = tf.unstack(arr)
rlist = []
for elm in alist:
    rlist.append(tf.sqrt(elm))
result2 = tf.stack(rlist)

diff = tf.reduce_sum(tf.abs(result1 - result2))

session = tf.Session()

print('The difference is %f ' % session.run(diff))

print('Tensorflow version is ' + tf.__version__)

# The output is 
'''
The difference is 0.001230 
Tensorflow version is 1.1.0
'''
#Time: 15:15pm, 4/28/2017






