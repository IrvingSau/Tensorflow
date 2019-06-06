import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

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

# Parameter setting
# Parameters setting
QPLS_NUM = 8;
WINDOW_SIZE = 9; # 9 day predict one day
FEATURE_NUM = 6; # HLOC + Volumn + Return
# NUM_INPUT_NEURONS = QPLS_NUM + WINDOW_SIZE * FEATURE_NUM; # 62
# NUM_HIDDEN_NEURONS = 30
# NUM_OUTPUT_NEURONS = 2

#### TEST PARAMETER SETTING
NUM_INPUT_NEURONS = 1
NUM_HIDDEN_NEURONS = 10
NUM_OUTPUT_NEURONS = 1

# 定义神经网络层
def add_Gaussian_layer(inputs, in_size, out_size, activation_function=None):
    ##### 定义权重和偏执
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size])) + 0.1 # Bias 不为0
    Wx_plus_bias = tf.matmul(inputs, Weight) + bias # 高斯函数输入

    # ##### 层正则化区域
    # # Batch Normalization
    # fc_mean, fc_var = tf.nn.moments(
    #     Wx_plus_bias,
    #     axes = [0]
    # )
    # scale = tf.Variable(tf.ones([out_size]))
    # shift = tf.Variable(tf.zeros([out_size]))
    # epsilon = 0.001
    # Wx_plus_b = tf.nn.batch_normalization(Wx_plus_bias, fc_mean, fc_var, shift, scale, epsilon)
    if activation_function is 1:
        ######    高斯函数定义区域
        # n_input = (self.input_data_trainX).shape[1]
        # n_output = (self.input_data_trainY).shape[1]
        n_input = 1;
        n_output = 1;

        # Weight = tf.Variable(tf.random_normal([NUM_HIDDEN_NEURONS, n_output]))
        # bias = tf.Variable(tf.zeros([1, n_output])) + 0.1  # Bias 不为0

        # 计算数据中心 c
        c = tf.Variable(tf.random_normal([NUM_HIDDEN_NEURONS, n_input]), name='c')
        # 计算方差
        variance = tf.Variable(tf.random_normal([1, NUM_HIDDEN_NEURONS]), name='variance')
        # 方差的平方
        variance_2 = tf.square(variance)
        # 计算特征样本与中心的距离
        dist = tf.reduce_sum(tf.square(tf.subtract(tf.tile(xs, [NUM_HIDDEN_NEURONS, 1]), c)), 1)
        dist = tf.multiply(1.0, tf.transpose(dist))
        # 高斯函数输出
        outputs = tf.exp(tf.multiply(-1.0, tf.divide(dist, tf.multiply(2.0, variance_2))))

    elif activation_function is 2:
        outputs = tf.nn.relu(Wx_plus_bias)
    elif activation_function is None:
        outputs = Wx_plus_bias;

    return outputs

def add_output_layer(RBF_OUT):
    n_input = 300;
    n_output = 300;

    Weight = tf.Variable(tf.random_normal([NUM_HIDDEN_NEURONS, n_output]))
    bias = tf.Variable(tf.zeros([1, n_output])) + 0.1  # Bias 不为0
    output_layer_in = tf.matmul(RBF_OUT, Weight) + bias

    ## 输出层输出
    y_pred = tf.nn.sigmoid(output_layer_in)

    return y_pred

# Create data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]      # 1个特征 300个训练样本
noise = np.random.normal(0, 0.05, x_data.shape)      # 为每个样本x增加噪音
y_data = np.square(x_data) - 0.5 + noise             # 设置我们的Y: 300行


# 定义输入神经网络的值的占位符 placeholder([num_examples, num_feature])
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# Hidden layer 1
Gaussian_Layer = add_Gaussian_layer(xs, in_size=NUM_INPUT_NEURONS, out_size=NUM_HIDDEN_NEURONS, activation_function=1)
# Output layer
prediction = add_output_layer(Gaussian_Layer)

# Define Loss function (MSE)
loss = tf.reduce_mean(          # 求平均值
    tf.reduce_sum(              # 求和
        tf.square(ys - prediction), reduction_indices=[1]))  # 差平方
        # reduction_indices 为对张量进行维度处理， 此处为1维（将多个不同的error相加）

# 定义每一个step中使用什么优化器来最小化loss
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

# 定义session

sess = tf.Session()

sess.run(init)


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()
for i in range(1000):
    # session run时传入输入值
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # try:
        #     ax.lines.remove(lines[0])  # 去除掉Lines的第一个线段
        # except Exception:
        #     pass

        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        # plt.pause(0.1)










