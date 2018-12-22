#!/usr/bin/env python3
"""
Gradients for ss_mat_mul.
"""
 
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
ssmatmul_mod = tf.load_op_library('./ssmatmul_op.so')
 
@ops.RegisterGradient("SSMatMul")
def _ss_mat_mul_grad_cc(op, grad):
    """
    The gradient for `ss_mat_mul` using the operation implemented in C++.
    
    :param op: `ss_mat_mul` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `ssmat_mul` op.
    :return: gradients with respect to the input of `inner_product`.
    """
    
    return ssmatmul_mod.ss_mat_mul_grad(grad, op.inputs[0], op.inputs[1]), tf.zeros(op.inputs[1].shape)
