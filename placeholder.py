import tensorflow as tf

# Place holder用于define输入，例如声明类型等
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)


# placeholder 占位符：作用是在session.run()时再为变量进行赋值
# 设置输入时使用字典的数据类型 （feed_dict{[], []}）


# 定义如何进行output的计算
output = tf.multiply(input1, input2)

# 不需要使用init进行初始化，因为使用了placeholder后，input1 和 input2 在sess.run()时初始化
# init = tf.initialize_all_variables()



with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [8.]}))

# 字典数据类型
# feed_dict = {key: value, key: value}
# 这里我们仍然是基于tensor进行计算 ： 1维张量 = 1个常数的矩阵



