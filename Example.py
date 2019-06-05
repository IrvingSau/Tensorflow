import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
# Produce data by numpy; define type as float32
x_data = np.random.rand(100).astype(np.float32)
# Construct the actual value of y
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
# 定义模型
# 定义参数的变量 （生成weight为一个数字[一维张量]， 范围-1到1）（但还没有初始化）
Weights = tf.Variable(tf.random.normal([1], -1.0,  1.0))
biases = tf.Variable(tf.zeros([1]))

# 预测的y
y = Weights * x_data + biases

# 定义损失函数 - Mean Square Error
loss = tf.reduce_mean(tf.square(y-y_data))

# 定义优化方式
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 使用optimizer最小化损失函数
train = optimizer.minimize(loss)

# 初始化模型
# 初始化变量
init = tf.global_variables_initizlizer()

# 通过session指定运行模型图中的哪一个部分
sess = tf.Session()
sess.run(init)

for step in range(200):
    # 每一次都运行一次训练器进行调参
    sess.run(train)
    # 每隔20步打印一下权重和偏执
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

### create tensorflow structure end ###
