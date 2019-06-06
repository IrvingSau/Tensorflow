import tensorflow as tf

# 一定要声明它是一个变量对象(tf.Variable())
state = tf.Variable(0, name= "counter")

# print(state.name)

one = tf.constant(1)

# 将new value来作为一个temp的记录器
new_value = tf.add(state, one)

# 声明如何update: 通过assign对state进行赋值new_value的操作
update = tf.assign(state, new_value)

# 初始化所有的变量 (初始化变量声明)
init = tf.global_variables_initializer()

# 激活初始化的变量 (通过构建session以及session)
with tf.Session() as sess:
    # 初始化变量实际的操作
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))


# 所有的获取值都需要session来完成