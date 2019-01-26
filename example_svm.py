import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# 鸢尾花数据
iris = datasets.load_iris()
# 取所有行，第1列第2列
# 为了方便绘图仅选择两个特征
X = iris.data[:, :2]
y = iris.target

# 测试样本（绘制分类区域）
# np.linspace(X,Y,N)在X和Y之间产生N个等间距的数列
xlist1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
xlist2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
# np.meshgrid()从一个坐标向量中返回一个坐标矩阵
# xlist1变成XGrid1的行向量,xlist2变成XGrid2的列向量
# XGrid1和XGrid2的维数是一样的，是网格矩阵也就是坐标矩阵
XGrid1, XGrid2 = np.meshgrid(xlist1, xlist2)

# 非线性SVM：RBF核，超参数为0.5，正则化系数为1，SMO迭代精度1e-5, 内存占用1000MB
svc = svm.SVC(kernel='rbf', C=1, gamma=0.5, tol=1e-5, cache_size=1000).fit(X, y)
# 预测并绘制结果
# ravel()函数是将矩阵变成一个一维的数组
Z = svc.predict(np.vstack([XGrid1.ravel(), XGrid2.ravel()]).T)
Z = Z.reshape(XGrid1.shape)
plt.contourf(XGrid1, XGrid2, Z, cmap=plt.cm.hsv)
plt.contour(XGrid1, XGrid2, Z, colors=('k',))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', linewidth=1, cmap=plt.cm.hsv)
plt.show()