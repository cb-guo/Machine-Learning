'''
支持向量机～简化版 SMO 算法
2018/6/20
'''

import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat 


# i 是第一个 alpha 的下标，m 是所有 alpha 的数目
# 只要函数值不等于输入值 i，函数就会进行随机选择
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j    


# 用于调整大于 H 或者小于 L 的 alpha 值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj        


# 5个输入参数分别为 dataMatIn=数据集，classLabels=类别标签，常数 C，toler=容错率 和 maxIter=退出前的最大循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn) # (100, 2)
    labelMat = np.mat(classLabels).transpose() # (100, 1)
    b = 0
    m, n = dataMatrix.shape # m = 100, n = 2
    alphas = np.mat(np.zeros((m, 1)))
    
    # iter 存储在没有任何alpha改变的情况下遍历数据集的次数，当该变量达到输入值maxIter时，函数结束运行并退出
    iter = 0
    while (iter < maxIter):
        # 每次循环中先将 alphaPairsChanged 设置为0，然后再对整个集合顺序遍历
        # 变量 alphaPairsChanged 用于记录 alpha 是否已经进行优化
        alphaPairsChanged = 0
        for i in range(m):
            # fXi 是我们预测的类别
            # alphas的shape = (100, 1), labelMat的shape=(100,1),multiply()再转置的shape = (1,100)
            # dataMatrix得到shape = (100,2)，dataMatrix[i:]的shape=(1,2),相乘之后shape=(100,1)
            # fXi的shape= (1,100)(100,1) + b = 一个数字
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            # Ei为对输入 xi 的预测值和真实输出值 yi 之差
            Ei = fXi - float(labelMat[i])
            # 如果 alpha 可以更改进入优化过程
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
               ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                    # 随机选择第二个 alpha
                    j = selectJrand(i, m)
                    fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                    Ej = fXj - float(labelMat[j])
                    # python中是引用传递，不适用copy方法，将看不到新旧数值的变化
                    alphaIoId = alphas[i].copy()
                    alphaJoId = alphas[j].copy()
                    #　保证 alpha 在　０～Ｃ 之间
                    if (labelMat[i] != labelMat[j]):
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    if L == H:
                        # print('L==H')
                        continue
                        
                    eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
                        dataMatrix[i,:] * dataMatrix[i,:].T - \
                        dataMatrix[j,:] * dataMatrix[j,:].T
                    # eta 是用于做分母的，不能为０
                    if eta >= 0:
                        print('eta >= 0')
                        continue
                    # 更新　alphas[j] 数值
                    alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                    alphas[j] = clipAlpha(alphas[j], H, L)
                    if (abs(alphas[j] - alphaJoId) < 0.00001):
                        # print('j is not moving enough')
                        continue
                    # 更新　alphas[i] 数值
                    alphas[i] += labelMat[j] * labelMat[i] * (alphaJoId - alphas[j])
                    b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIoId) * \
                        dataMatrix[i,:] * dataMatrix[i,:].T - \
                        labelMat[j] * (alphas[j] - alphaJoId) * \
                        dataMatrix[i,:] * dataMatrix[j,:].T
                    b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIoId) * \
                        dataMatrix[i,:] * dataMatrix[j,:].T - \
                        labelMat[j] * (alphas[j] - alphaJoId) * \
                        dataMatrix[j,:] * dataMatrix[j,:].T
                    if (0 < alphas[i]) and (C > alphas[i]):
                        b = b1
                    elif (0 < alphas[j]) and (C > alphas[j]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
                    # print('iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        # print('iteration number: %d' % iter)
    return b, alphas


# w 的计算
def calcWs(alphas, dataArr, labelArr):
    X = np.mat(dataArr)     # (100, 2)
    labelMat = np.mat(labelArr).transpose() #(100, 1)
    m, n = np.shape(X)      # m = 100, n = 2
    w = np.zeros((n, 1))    # (100, 1)
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w


# 画出完整分类图
def plotFigure(weights, b, alphas):
    x, y = loadDataSet('testSet.txt')
    xarr = np.array(x)
    n = np.shape(x)[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in np.arange(n):
        if int(y[i]) == 1:
            x1.append(xarr[i,0]); y1.append(xarr[i,1])
        else:
            x2.append(xarr[i,0]); y2.append(xarr[i,1])
    
    plt.scatter(x1, y1, s = 30, c = 'r', marker = 's')
    plt.scatter(x2, y2, s = 30, c = 'g')
    
    # 画出 SVM 分类直线
    xx = np.arange(0, 10, 0.1) 
    # 由分类直线 weights[0] * xx + weights[1] * yy1 + b = 0 易得下式
    yy1 = (-weights[0] * xx - b) / weights[1]
    # 由分类直线 weights[0] * xx + weights[1] * yy2 + b + 1 = 0 易得下式
    yy2 = (-weights[0] * xx - b - 1) / weights[1]
    # 由分类直线 weights[0] * xx + weights[1] * yy3 + b - 1 = 0 易得下式
    yy3 = (-weights[0] * xx - b + 1) / weights[1]
    plt.plot(xx, yy1.T)
    plt.plot(xx, yy2.T)
    plt.plot(xx, yy3.T)
    
    # 画出支持向量点
    for i in range(n):
        if alphas[i] > 0.0:
            plt.scatter(xarr[i,0], xarr[i,1], s = 150, c = 'none', alpha = 0.7, linewidth = 1.5, edgecolor = 'red')

    plt.xlim((-2, 12))
    plt.ylim((-8, 6))
    plt.show()


# 主函数
if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('/home/gcb/data/testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40) 
    w = calcWs(alphas, dataArr, labelArr)
    plotFigure(w, b, alphas)
    print(b)
    print(alphas[alphas > 0]) # 支持向量对应的 alpha > 0
    print(w)