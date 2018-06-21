
'''
支持向量机～复杂数据上应用核函数
2018/6/20
'''

import numpy as np

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


# 核转换函数
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        # 元素间的除法
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- \
        That Kernel is not recognized')
    return K


# 建立一个数据结构来保存所有的重要值，这样较为便利
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 误差缓存,第一列为是否有效标志位，第二列为实际的Ｅ值
        self.eCache = np.mat(np.zeros((self.m, 2))) 
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i,:], kTup)


# 计算并返回 E 值
def calcEk(oS, k):
    # 预测值
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.K[:, k])) + oS.b
    # 误差值
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 内循环中的启发式方法
# 用于选择第二个 alpha 或者说内循环的　alpha 值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 计算误差值并存入缓存中，在对alpha值进行优化之后会用到这个值
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 完整 Platt SMO 算法中的优化例程
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
       ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0) ):
            # 用于选择第二个 alpha 或者说内循环的　alpha 值
            j, Ej = selectJ(i, oS, Ei)
            alphaIoId = oS.alphas[i].copy()
            alphaJoId = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                # print('L==H')
                return 0
            
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
            if eta >= 0:
                # print('eta >= 0')
                return 0
            
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
            updateEk(oS, j) # 更新误差缓存
            
            if (abs(oS.alphas[j] - alphaJoId) < 0.00001):
                # print('j not moving enough')
                return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJoId - oS.alphas[j])
            updateEk(oS, i) # 更新误差缓存
            
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIoId) * \
                oS.K[i, i] - oS.labelMat[j] * \
                (oS.alphas[j] - alphaJoId) * oS.K[i, j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIoId) * \
                oS.K[i, j] - oS.labelMat[j] * \
                (oS.alphas[j] - alphaJoId) * oS.K[j, j]
                
            if (0 < oS.alphas[i]) and (oS.alphas[i] < oS.C):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.alphas[j] < oS.C):
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
    else:
        return 0


# 完整版 SMO 算法中的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):  # 遍历所有的值
                alphaPairsChanged += innerL(i, oS)
                # print('fullSet, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 遍历非边界值
            nonBoundIs = np.nonzero((0 < oS.alphas.A) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print('non-bound, iter: %d i: %d, pairs changed %d' %(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print('iteration number: %d' % iter)
    return oS.b, oS.alphas



# 利用核函数进行分类的径向基测试函数
# 第一个 for 和第二个 for 仅仅数据集不同，前者训练集后者测试集
def testRbf(k1 = 1.3):
    # 用训练集训练模型
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas =  smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % np.shape(sVs)[0])
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' %(float(errorCount) / m))
    
    # 用训练好的模型在测试集计算错误率
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the best error rate is: %f' %(float(errorCount) / m))


# 主函数
if __name__ == '__main__':
    testRbf()
