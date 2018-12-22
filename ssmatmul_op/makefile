TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

all:
	g++ -std=c++11 -shared ssmatmul_op.cc ssmatmul_grad.cc -o ssmatmul_op.so -fPIC -I ${TF_INC} -I ${NSYNC} -O2 -L${TF_LIB} -l:libtensorflow_framework.so
#
#
#all:
#	g++ -std=c++11 -shared inner_product_op.cc -o inner_product_op.so -fPIC -I ${TF_INC} -I ${NSYNC} -O2 -L${TF_LIB} -l:libtensorflow_framework.so
#	g++ -std=c++11 -shared inner_product_grad.cc -o inner_product_grad.so -fPIC -I ${TF_INC} -I ${NSYNC} -O2 -L${TF_LIB} -l:libtensorflow_framework.so
