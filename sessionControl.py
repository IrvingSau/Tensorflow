import tensorflow as tf

# 定义两个张量
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                        [2]])

# 定义矩阵的乘法
productOp = tf.matmul(matrix1, matrix2)

# Method 1
sess = tf.Session()

result = sess.run(productOp)

print(result)
# output = [[12]]

# Method 2
# 声明session 方法， 使用with as， 将在with 语句内不断使用sess
with tf.Session() as sess:
    result2 = sess.run(productOp)
    print(result)
