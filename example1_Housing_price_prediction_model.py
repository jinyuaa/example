import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from mpl_toolkits import mplot3d

'''
# 单变量房价预测模型
sns.set(context="notebook", style="whitegrid", palette="dark")

df0 = pd.read_csv('data0.csv', names=['square', 'price'])
sns.lmplot('square', 'price', df0, height=6, fit_reg=True)
plt.show()
'''

# 多变量房价预测模型
df1 = pd.read_csv('data1.csv', names=['square', 'bedrooms', 'price'])


# 数据规范化
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


df1 = normalize_feature(df1)
'''
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.scatter3D(df1['square'], df1['bedrooms'], df1['price'], c=df1['price'], cmap='Greens')
# plt.show()
'''
# 数据处理：添加 ones 列
ones = pd.DataFrame({'ones': np.ones(len(df1))})
df = pd.concat([ones, df1], axis=1)
# print(df.head())

# 数据处理：获取 x 和 y
X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)

print(y_data.shape)
# 创建线性回归模型(数据流图)
# 定义参数
alpha = 0.01   # 学习率
epoch = 500    # 训练全量数据的轮数

# 名字作用域和抽象节点 使数据流图更规整清晰（相比之前-0 的凌乱）

# 创建线性回归模型(即数据流图)
# 输入 X 的形状为 [47,3]
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, X_data.shape, name='X')
    # 输出 y 的形状为 [47,1]
    y = tf.placeholder(tf.float32, y_data.shape, name='y')

with tf.name_scope('hypothesis'):
    # 权重变量 W 形状 [3,1]
    W = tf.get_variable("weights", (X_data.shape[1], 1), initializer=tf.constant_initializer())
    # 假设函数 h(x)=w0*x0+w1*x1+w2*x2 其中x0恒为1
    # 推理值 y_pred 形状为[47,1]
    y_pred = tf.matmul(X, W, name='y_pred')

with tf.name_scope('loss'):
    # 损失函数采用最小二乘法 (y_pred - y) 是形如[47,1]的向量
    # tf.matmul(a, b, transpose_a=True) 表示矩阵a的转置乘矩阵b,即[1,47]x[47,1]
    # 实现了 作差 求和 乘以 1/2n,即损失函数
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred-y), (y_pred-y), transpose_a=True)

with tf.name_scope('train'):
    # 随机梯度下降优化器
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    # 单轮训练操作
    train_op = opt.minimize(loss_op)

# 创建会话(运行环境)
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 创建 FileWriter 实例，并传入当前会话加载的数据流
    writer = tf.summary.FileWriter('./summary/linear-regression-1', sess.graph)
    # 记录所有损失值
    loss_data = []
    # 开始训练模型
    # 因为训练集较少，所以采用梯度下降优化算法，每次都使用全量数据训练
    for step in range(1, epoch+1):
        _, loss, w = sess.run([train_op, loss_op, W], feed_dict={X: X_data, y: y_data})
        # 记录每一轮损失值变化情况
        loss_data.append(float(loss))
        if step % 10 == 0:
            # loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
            log_str = "Epoch %d \t Loss = %.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g"
            print(log_str % (step, loss, w[1], w[2], w[0]))

# 关闭 FileWriter 的输出流
writer.close()

# Epoch 500  Loss = 0.132  Model: y = 0.8304x1 + 0.0008239x2 + 4.138e-09

# 可视化损失值
sns.set(context="notebook", style="whitegrid", palette="dark")
ax = sns.lineplot(x='epoch', y='loss', data=pd.DataFrame({'loss': loss_data, 'epoch': np.arange(epoch)}))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
plt.show()
