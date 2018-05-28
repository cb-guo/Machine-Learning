
'''
2018/5/28
从疝气病症预测病马的死亡率

'''

import numpy as np
import matplotlib.pyplot as plt

# 预测类别
def classifyVector(x, weights):
    prob = sigmoid(np.sum(x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# 读取文件
def colicTest():
    # 读取训练集
    frTrain = open('/home/gcb/data/horseColicTraining.txt')
    # 读取测试集
    frTest = open('/home/gcb/data/horseColicTest.txt')
    
    # 定义存放 feature 和 label 的容器
    trainingSet = []; trainingLabels = []
    # 一次取文本中的一行
    # 一行共计 22 列，前 21 列为 feature， 最后一列为 label
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in np.arange(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    
    # 调用改进的随机梯度上升算法计算权重
    trainWeights = randgradAscent1(np.array(trainingSet), np.array(trainingLabels), 500)
    
    # 测试
    # 一行共计 22 列，前 21 列为 feature， 最后一列为 label
    # 测试错误计数
    errorCount = 0
    # 测试集总共样本数
    numTestVec = 0.0
    # 一次取出一行
    for line in frTest.readlines():
        # 测试集样本计数
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in np.arange(21):
            lineArr.append(float(currLine[i]))
        # 若预测和真实类别不符合，则错误计数加 1
        if int(classifyVector(np.array(lineArr), trainWeights) != int(currLine[21])):
            errorCount += 1
    
    # 计数这一次的错误率
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate )
    return errorRate

      
# 定义 sigmoid 函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(- x))


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
     # 共计测试 10 次，即测试集读取10次，每次算出错误率，最后算出平均错误率
    numTests = 10
    # 10 次中，每一次的错误率
    errorSum = 0.0
    for k in np.arange(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" %(numTests, errorSum / float(numTests)))