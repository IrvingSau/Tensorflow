import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import math
import matplotlib.pyplot as plt

# 如何定义一个神经层
    # 神经层中有什么？
    # 1. Bias
    # 2. Weight 矩阵
    # 3. Weight * （Input + Bias ）的求和
    # 4. Activation function以及输出

# 如何定义一个神经网络
    # 1. 定义Layer（input, in_size, out_size, activation_function）
    # 2. 定义placeholder(tf.float32, 输入的格式 [num_examples, num_features])
    # 3. 定义模型架构（Hidden layer + output layer）
    # 3. 定义Loss
    # 4. 定义每一个train_step做什么
    # 5. 初始化所有变量
    # 6. 定义session
    # 7. 用for来执行train
        # i. sess.run(train_step, feed_dict{x:[], y:[]})

# 定义自己的激活函数
def gaussian(x):
    # print(x)
    return math.exp(- (x*x) / (0.25))

np_gaussian = np.vectorize(gaussian)
np_gaussian_32 = lambda x: np_gaussian(x).astype(np.float32)

# 定义该激活函数的一次梯度函数
def d_gaussian(x):
    # print(x)
    return (-8) * x * math.exp(- (x*x) / (0.25))
    # r = x % 1
    # if r <= 0.5:
    #     return 1
    # else:
    #     return 0
np_d_gaussian = np.vectorize(d_gaussian)

np_d_gaussian_32 = lambda x: np_d_gaussian(x).astype(np.float32)

def tf_d_gaussian(x,name=None):
    with tf.name_scope(name, "d_gaussian", [x]) as name:
        y = tf.py_func(np_d_gaussian_32,
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

def gaussiangrad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_gaussian(x)
    return grad * n_gr

def gaussiangrad2(op, grad):
    x = op.inputs[0]
    r = tf.mod(x,1)
    n_gr = tf.to_float(tf.less_equal(r, 0.5))
    return grad * n_gr

def tf_gaussian(x, name=None):
    with tf.name_scope(name, "gaussian", [x]) as name:
        y = py_func(np_gaussian_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=gaussiangrad)  # <-- here's the call to the gradient
        return y[0]


# 定义神经网络层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定义权重和偏执
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size])) + 0.1 # Bias 不为0

    # Wx_plus_bias = tf.matmul(inputs, Weight) + bias
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
        outputs = tf_gaussian(Wx_plus_bias)
    return outputs

# Create data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]      # 1个特征 300个训练样本
noise = np.random.normal(0, 0.05, x_data.shape)      # 为每个样本x增加噪音
y_data = np.square(x_data) - 0.5 + noise             # 设置我们的Y: 300行

# 定义输入神经网络的值的占位符 placeholder([num_examples, num_feature])
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# Hidden layer 1
il = add_layer(xs, in_size=1, out_size=10, activation_function=1)
# Hidden layer 2
hl = add_layer(il, in_size=10, out_size=10, activation_function=1)
# Output layer
prediction = add_layer(hl, in_size=10, out_size=1, activation_function=None)

# Define Loss function (MSE)
loss = tf.reduce_mean(          # 求平均值
    tf.reduce_sum(              # 求和
        tf.square(ys - prediction), reduction_indices=[1]))  # 差平方
        # reduction_indices 为对张量进行维度处理， 此处为1维（将多个不同的error相加）

# 定义每一个step中使用什么优化器来最小化loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

# 定义session

sess = tf.Session()

sess.run(init)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
for i in range(1000):
    # session run时传入输入值
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 1 == 0:
        try:
            ax.lines.remove(lines[0])  # 去除掉Lines的第一个线段
        except Exception:
            pass

        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        plt.pause(0.1)


plt.pause(10)







