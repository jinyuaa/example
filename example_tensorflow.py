import tensorflow as tf
import numpy as np

# 构造一个满足一元二次函数 y=ax^2+b 的原始数据
# 再构造一个最简单的神经网络，仅包含一个输入层、一个隐藏层、一个输出层
# 通过 Tensorflow 将隐藏层和输出层的 weights 和 biases 学习出来，看随训练次数增加，损失值是否再减小

# 1、加载数据及定义

# 首先生成输入数据，假设最后要学习的方程为 y=x^2 - 0.5
# 构造满足这个方程的一堆 x 和 y ，同时加入一些不满足方程的噪声点

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 加入噪声点，使它与 x_data 的维度一致，并拟合为均值为0，方差为0.05的正太分布
noise = np.random.normal(0, 0.05, x_data.shape)

y_data = np.square(x_data) - 0.5 + noise
# 定义 x 和 y 的占位符来作为将要输入神经网络的变量
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 2、构建网络

def add_layer(inputs, in_size, out_size, activation_function=None):
    # 构建权重 in_size x out_size 大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 构建偏置 1 x out_size 的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 矩阵相乘
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    # 得出输出数据
    return outputs

#构建隐藏层 假设有20个神经元
h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
# 构建输出层，假设输出层和输入层一样，有1个神经元
prediction = add_layer(h1, 20, 1, activation_function=None)

# 需要构建损失函数，计算误差，对二者差的平方求和再取平均，得到loss函数
# 运用梯度下降法，以0.1的学习速率最小化损失

# 计算预测值和真实值间的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 3、训练模型

# 让Tensorflow训练10000次，每50次输出训练的损失值
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):  # 训练1000次
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:    # 每50次打印一次损失值
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))