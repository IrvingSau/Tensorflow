import tensorflow as tf
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# 读取CRBF函数结果
# Read CRBF function result
functionResultSet = "CRBF_Function"
functionResult = pd.read_csv(functionResultSet, header=None)
functionResult = functionResult.values

# Read CRBF function derivation
functionDerivationSet = "CRBF_derivation"
functionDerivation = pd.read_csv(functionDerivationSet , header=None)
functionDerivation = functionDerivation.values

#   自定义CRBF函数
# Take value from 0 to 1 (with 3 digital fraction)
def CRBF(x):
    # x = math.tanh(x)
    # print('1x = ', x)
    # should be 1998
    rowIndex = int(x * 1998)
    # if out of the bound;
    if rowIndex < 0:
        return 1.7169e-005
    elif rowIndex > 1998:
        return 1.7169e-005
    # print(rowIndex)
    # Random Column index (0 - 99)
    columnIndex = random.randint(0, 99)

    # Query for result dataframe(input)
    result = functionResult[rowIndex][columnIndex]
    #print("result = ", result)
    return result


np_CRBF = np.vectorize(CRBF)
np_CRBF_32 = lambda x: np_CRBF(x).astype(np.float32)

# 先用最简单的tanh基本倒数
def d_CRBF(x):
    if x < 0.5:
        return 1 - pow((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)), 2)
    else:
        return -1 + pow((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)), 2)

# #　尝试使用CRBF倒数
# def d_CRBF(x):
#     x = math.tanh(x)
#     #print("2x = ", x)
#     # should be 1998
#     rowIndex = int(x * 1998)
#     # Query for result dataframe(input)
#     derivation = functionDerivation[rowIndex]
#     #print("derivation = ", derivation)
#     return derivation

np_d_CRBF = np.vectorize(d_CRBF)

np_d_CRBF_32 = lambda x: np_d_CRBF(x).astype(np.float32)

def tf_d_CRBF(x,name=None):
    with tf.name_scope(name, "d_CRBF", [x]) as name:
        y = tf.py_func(np_d_CRBF_32,
                       [x],
                       [tf.float32],
                       name=name,
                       stateful=False)
        return y[0]

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def CRBFgrad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_CRBF(x)
    return grad * n_gr

def CRBFgrad2(op, grad):
    x = op.inputs[0]
    r = tf.mod(x,1)
    n_gr = tf.to_float(tf.less_equal(r, 0.5))
    return grad * n_gr

def tf_CRBF(x, name=None):
    with tf.name_scope(name, "CRBF", [x]) as name:
        y = py_func(np_CRBF_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=CRBFgrad)  # <-- here's the call to the gradient
        return y[0]



# 定义神经网络层
def add_CRBF_layer(inputs, in_size, out_size, activation_function=None):

    # 定义权重和偏执
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size])) + 0.1 # Bias 不为0

    Wx_plus_bias = tf.matmul(inputs, Weight) + bias

    # Batch Normalization
    fc_mean, fc_var = tf.nn.moments(
        Wx_plus_bias,
        axes = [0]
    )
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    Wx_plus_bias = tf.nn.batch_normalization(Wx_plus_bias, fc_mean, fc_var, shift, scale, epsilon)

    # 如果激活函数为空，直接将Wx + b输出
    # 否则将其传入activation_function中
    if activation_function is None:
        outputs = Wx_plus_bias
    elif activation_function is 1:
        outputs = tf_CRBF(Wx_plus_bias)
    else:
        outputs = tf.nn.relu(Wx_plus_bias)
    return outputs

######################  DEFINE GRAPH

#########   DATA READING

# Create data
x_data = np.linspace(0, 1, 300)[:, np.newaxis]      # 1个特征 300个训练样本 0-1之间
# print(x_data)
noise = np.random.normal(0, 0.05, x_data.shape)      # 为每个样本x增加噪音
y_data = np.power(x_data, 2) - 0.5 + noise             # 设置我们的Y: 300行

######################

######################   Read CRBF

# 定义输入神经网络的值的占位符 placeholder([num_examples, num_feature])
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# input layer 1
h1 = add_CRBF_layer(xs, in_size=1, out_size=10, activation_function=None)
# Batch Normalization Layer
bn1 = tf.layers.batch_normalization(h1, training=True, name='bn1')
# # Hidden layer 2
h2 = add_CRBF_layer(h1, in_size=10, out_size=10, activation_function=1)
# Batch Normalization Layer
# bn2 = tf.layers.batch_normalization(h2, training=True, name='bn2')
# # # Hidden layer 3
h3 = add_CRBF_layer(h2, in_size=10, out_size=10, activation_function=1)
# # Batch Normalization Layer
# bn3 = tf.layers.batch_normalization(h3, training=True, name='bn3')
# # Hidden layer 4
h4 = add_CRBF_layer(h3, in_size=10, out_size=10, activation_function=1)

# # Output layer
prediction = add_CRBF_layer(h4, in_size=10, out_size=1, activation_function=None)

# Define Loss function (MSE)
loss = tf.reduce_mean(          # 求平均值
    tf.reduce_sum(              # 求和
        tf.square(ys - prediction), reduction_indices=[1]))  # 差平方
        # reduction_indices 为对张量进行维度处理， 此处为1维（将多个不同的error相加）

# 定义每一个step中使用什么优化器来最小化loss
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

# 定义session

sess = tf.Session()

sess.run(init)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# X data test

for i in range(1000):
    # session run时传入输入值
    # print(sess.run(bn0, feed_dict={xs: x_data})) # same with xs
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])  # 去除掉Lines的第一个线段
        except Exception:
            pass

        print('loss = ', sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        plt.pause(0.1)


plt.pause(10)