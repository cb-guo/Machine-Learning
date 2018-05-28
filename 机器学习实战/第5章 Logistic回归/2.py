
'''
2018/5/28
随机梯度上升优化算法
'''

import numpy as np
import matplotlib.pyplot as plt

# 从文本中读取数据集
def loadDataSet():
    x = []
    y = []
    fr = open('/home/gcb/data/testSet.txt')
    for line in fr.readlines():
        # line.strip() 截取掉所有回车字符(即每行最后一个字符)
        # split(str="")  str - 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等分隔
        line_x = line.strip().split()
        # 此处第一列的 1，即为biases
        x.append([1.0, float(line_x[0]), float(line_x[1])])
        y.append([int(line_x[2])])
    return np.mat(x),np.mat(y)

# 读进数据集之后，现在让我们来以图像化显示一下数据集
def plotFigure():
    x, y = loadDataSet()
    xarr = np.array(x)
    n = np.shape(x)[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in np.arange(n):
        if int(y[i]) == 1:
            x1.append(xarr[i,1]); y1.append(xarr[i,2])
        else:
            x2.append(xarr[i,1]); y2.append(xarr[i,2])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # scatter 画出散点图
    ax.scatter(x1, y1, s = 30, c = 'r', marker = 's')
    ax.scatter(x2, y2, s = 30, c = 'g')
    plt.show()

# 定义 sigmoid 函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(- x))

# 画出决策边界
# 用此函数来画出三种梯度上升算法的分类直线
def plotBestFit(weights):
    x, y = loadDataSet()
    xarr = np.array(x)
    n = np.shape(x)[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in np.arange(n):
        if int(y[i]) == 1:
            x1.append(xarr[i,1]); y1.append(xarr[i,2])
        else:
            x2.append(xarr[i,1]); y2.append(xarr[i,2])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # scatter 画出散点图
    ax.scatter(x1, y1, s = 30, c = 'r', marker = 's')
    ax.scatter(x2, y2, s = 30, c = 'g')
    
    # 画出Logistic 分类直线
    a = np.arange(-3.0, 3.0, 0.1) # (60,)
    # 由分类直线 weights[0] + weights[1] * a + weights[2] * b = 0 易得下式
    b = (-weights[0] - weights[1] * a) / weights[2]
    # print(b.shape)   # (1, 60)
    # print(b.T.shape) # (60, 1)
    ax.plot(a, b.T)
    plt.title('BestFit')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def randgradAscent(x, y):
    m, n = np.shape(x) # 100 3
    alpha = 0.01
    weights = np.ones(n) #(3,)  [1. 1. 1.]
    for i in np.arange(m):
        h = sigmoid(np.sum(x[i] * weights))
        error = y[i] - h
        weights = weights + alpha * error * x[i]
    return (np.mat(weights)).T


# 程序入口
if __name__ == '__main__':
    x, y = loadDataSet()
    weights = randgradAscent(np.array(x), np.array(y))
    print(weights)
    plotBestFit(weights)