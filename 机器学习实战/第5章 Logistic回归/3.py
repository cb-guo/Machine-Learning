
'''
2018/5/28
改进的随机梯度上升算法
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

# 改进的随机梯度上升算法
def randgradAscent1(x, y, cycle = 150):
    m, n = np.shape(x) # 100 3
    weights = np.ones(n) # (3,) [1. 1. 1.]
    # 循环 150 次
    for j in np.arange(cycle):
        dataindex = np.arange(m) # 100
        # 在 150 次之内，每一次又循环 100 次
        for i in np.arange(m):
            # 定义学习率，随着 i,j 的增大，学习率越来越小，0.01保证学习率永远不为0
            alpha = 4 / (1.0 + j + i) + 0.01
            # 从 0 - len(dataindex) 中随机取出一个数
            randindex = int(np.random.uniform(0, len(dataindex)))
            # 预测类别的概率
            h = sigmoid(np.sum(x[randindex] * weights))
            # 误差
            error = y[randindex] - h
            # 梯度上升更新权重
            weights = weights + alpha * error * x[randindex]
            # dataindex 原来为 array ，转化为 list
            dataindex = dataindex.tolist()
            # 删除已取出的数
            del(dataindex[randindex])
            # dataindex 再次转化为 array
            dataindex = np.array(dataindex)
    # 注意，此处开始创建 weights 的shape为 (3,) ，现在把weight转化为matrix类型，再者转置，化为 (3, 1)
    return np.mat(weights).T

# 程序入口
if __name__ == '__main__':
    x, y = loadDataSet()
    weights = randgradAscent1(np.array(x), np.array(y))
    print(weights)
    plotBestFit(weights)