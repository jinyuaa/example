#  MNIST-softmax模型去做手写体数字识别 使用低层的API 直接定义网络结构里面每一个模型参数

from __future__ import print_function
import tensorflow as tf

# 导入 MNIST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('Mnist_data/', one_hot=True)

# 定义超参数
learning_rate = 0.1
num_steps = 500      # 训练500步
batch_size = 128     # 每一步的训练就是128张图片进行训练
display_step = 100   # 展示100步

# 神经网络参数
n_hidden_1 = 256  # 第一层神经元个数
n_hidden_2 = 256  # 第二层神经元个数
num_input = 784   # MNIST 输入数据（图像大小：28*28）转换成1维数组
num_classes = 10  # MNIST 手写体数字类别（0-9）类别

# 输入到数据流图中的训练数据
# 其实是定义了一个二维数组，但是有一维为 None ,意为是可变长的一个动态数组
X = tf.placeholder("float", [None, num_input])  # 784维的一阶数组
Y = tf.placeholder("float", [None, num_classes])

# 权重和偏置
weights = {
    # tf.random_normal  用于从服从指定正态分布的数值中取出指定个数的值
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes])),
}


# 定义神经网络
def neural_net(x):
    # 第一层隐藏层（256个神经元） layer_1 =  X * W + b  第一层隐藏层的输出结果
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # 第二层隐藏层（256个神经单元）
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # 输出层
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# 构建模型
# X: 1282x784 的数组; logits:128x10 的数组
logits = neural_net(X)


# 定义损失函数和优化器
# 损失函数用 softmax_cross_entropy（softmax 、交叉熵）
# logits是算出的预测的数值，Y是训练数据集的标签，取出的真实结果
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# 通过梯度下降的方式进行不断优化
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# 定义预测准确率
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有变量（赋默认值）
init = tf.global_variables_initializer()

# 开始训练
# 会话Session() ：用 with 语句实现一个上下文环境的
with tf.Session() as sess:
    # 执行初始化操作
    sess.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 执行训练操作，包括前向和后向传播
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # 计算损失值和准确率
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step" + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    print("Optimization Finished!")

    # 计算测试数据的准确率
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


# 在特定的超参数设定输入下，梯度下降法只找到局部最优解