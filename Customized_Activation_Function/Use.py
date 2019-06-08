import numpy as np
import tensorflow as tf
# -*- encoding:utf-8 -*-
# !/usr/local/env python

import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops


def gaussian(x):
    return math.exp(- (x * x) / (0.25))


def gaussian_grad(x):
    return (-8) * x * math.exp(- (x * x) / (0.25))


gaussian_np = np.vectorize(gaussian)
gaussian_grad_np = np.vectorize(gaussian_grad)

gaussian_np_32 = lambda x: gaussian_np(x).astype(np.float32)
gaussian_grad_np_32 = lambda x: gaussian_grad_np(x).astype(np.float32)


def gaussian_grad_tf(x, name=None):
    with ops.name_scope(name, "gaussian_grad_tf", [x]) as name:
        y = tf.py_function(gaussian_grad_np_32, [x], [tf.float32], name=name, stateful=False)
        return y[0]


def my_py_func(func, inp, Tout, stateful=False, name=None, my_grad_func=None):
    # need to generate a unique name to avoid duplicates:
    random_name = "PyFuncGrad" + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(my_grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": random_name, "PyFuncStateless": random_name}):
        return tf.py_function(func, inp, Tout, name=name)


def _gaussian_grad(op, pred_grad):
    x = op.inputs[0]
    cur_grad = gaussian_grad(x)
    next_grad = pred_grad * cur_grad
    return next_grad


def gaussian_activation(x, name=None):
    with ops.name_scope(name, "gaussian_activator", [x]) as name:
        y = my_py_func(gaussian_np_32,
                       [x],
                       [tf.float32],
                       stateful=False,
                       name=name,
                       my_grad_func=_gaussian_grad)
    return y[0]


a = tf.constant([2, 5])

with tf.Session() as sess:
    b = gaussian_activation(a)
    print(sess.run(b))

