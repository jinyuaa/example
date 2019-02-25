# 用 keras 实现 CNN

from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
import os
import tensorflow.gfile as gfile
from keras.models import load_model

#  加载 MNIST 数据集
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist\mnist.npz')
# print(x_train.shape, type(x_train))

# 数据处理: 规范化
img_rows, img_cols = 28, 28
# 本数据集为灰度图片通道数为1 当涉及RGB格式则不同，一般使用channels_last格式
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = [1, img_rows, img_cols]
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = [img_rows, img_cols, 1]

# print(x_train.shape, type(x_train))

# 将数据类型转换成 float32
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
# 数据归一化 [0,1]
X_train /= 255
X_test /= 255

# print(X_train.shape[0])

# 统计训练数据中各标签数量
label, count = np.unique(y_train, return_counts=True)
# print(label, count)
fig = plt.figure()
plt.bar(label, count, width=0.7, align='center')
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(label)
plt.ylim(0, 7500)
# 显示表示每个柱状图具体值
for a, b in zip(label, count):
    # ha(horizontal alignment)水平对齐方式; va(vertical alignment)垂直对齐方式
    plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
plt.show()

# one-hot 编码
n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print(y_train[2])   # 4
print(Y_train[2])   # [0 0 0 0 1 0 0 0 0 0]

# 使用 Keras sequential model 定义 MNIST CNN 网络
model = Sequential()
## Feature Extraction
# 第1层卷积，32个3x3的卷积核，激活函数使用relu
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# 第2层卷积，64个3x3的卷积核，激活函数使用relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# 最大池化层，池化窗口为2x2
model.add(MaxPool2D(pool_size=(2, 2)))
# 将 Pooled Feature map 摊平后输入全连接网络
model.add(Flatten())

## Classification
# 全连接层
model.add(Dense(128, activation='relu'))
# Dropout 50% 的输入神经元
model.add(Dropout(0.5))
# 使用 softmax 激活函数做多分类，输出各数字的概率
model.add(Dense(n_classes, activation='softmax'))
# 查看 MNIST CNN 模型网络结构
model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# 训练模型，将指标保存到 history 中
history = model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=2, validation_data=(X_test, Y_test))

# 可视化指标
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

plt.show()

# 保存模型 参数及结构
save_dir = 'D:/PycharmProject/example/model1/'
if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)

model_name = 'keras_mnist1.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s' % model_path)

#加载模型
mnist_model = load_model(model_path)
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss: {}".format(loss_and_metrics[0]))
print("Test Accuracy: {}".format(loss_and_metrics[1]*100))

predicted_classes = mnist_model.predict_classes(X_test)
# np.nonzero
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print("Classified correctly count : {}".format(len(correct_indices)))
print("Classified incorrectly count : {}".format(len(incorrect_indices)))

# CNN准确率到达 99.13%(浮动)
# 而 softmax 两层全连接网络 准确率到达 97.89%(浮动)