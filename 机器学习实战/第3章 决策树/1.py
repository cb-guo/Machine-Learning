
'''
2018-5-30
决策树
'''
import numpy as np
from math import log

def createDataSet():
    dataSet = [
              [1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    # 数据集样本数
    numEntries = len(dataSet)
    # 定义字典，用于计数
    labelCounts = {}
    # 从数据集中，每次取出一行
    for featVec in dataSet:
        # 取出每一行的最后一列，即 'yes' or 'no'
        currentLabel = featVec[-1]
        # 判断 'yes' or 'no' 是否在字典中，不在加入计数为0，在则计数加1
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 定义容器，存放 熵
    shannonEnt = 0.0
    # 依据信息熵公式，计算该 dataSet 的信息熵 
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    # 返回 dataSet 的信息熵
    return shannonEnt

# 按照给定特征划分数据集
# dataSet 数据集
# axis　　列号
# value　将列号为axis，值为value的　其他数据分个出来，看实例
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] 
            reducedFeatVec.extend(featVec[axis + 1 : ])
            retDataSet.append(reducedFeatVec)
    return retDataSet 

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 每一行最后一列为 label,计算 feature 列数
    numFeatures = len(dataSet[0]) - 1
    #计算整个数据集的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # bestInfoGain　信息增益预先设为为 0.0
    # bestFeature 最好划分的列标签，预先设为 -1
    bestInfoGain = 0.0; bestFeature = -1
    # 依据 feature 数进行循环
    for i in np.arange(numFeatures):
        # 将每一行数据取出，存为list，次数为feature数，即没有最后一列label
        featList = [example[i] for example in dataSet]
        # set 去冗余
        uniqueVals = set(featList)
        # 定义熵容器
        newEntropy = 0.0
        # 从 uniqueVals 集合中迭代
        for value in uniqueVals:
            # 从每一列开始，按值划分数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 见公式
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) 
        infoGain = baseEntropy - newEntropy
        # 找到信息增益最大的列号
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 返回信息增益最大的列号
    return bestFeature


# 如果数据集已经处理了所有属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点，
# 在这种情况下，我们通常会采用 多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    # 定义一个用于计数的字典
    # 注，此时classList 只有一列，为类标签，因为类标签不唯一，才用此方法找最多的label
    classCount = {}
    # 从 classList 迭代取值
    for vote in classList:
        # 如果从classList中取出的值不在classCount字典中，则将该值放入字典，计数为1，否则在字典中的该值计数加1
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    
    # 找到字典中 value 最大的 key 并返回
    newvalue = -1
    for key in classCount:
        if newvalue < classCount[key]:
            newkey = key
            newvalue = classCount[key]
    return newkey

# 创建树的函数代码
def createTree(dataSet, labels): # 两个输入参数-- 数据集， 标签列表
    # 将 dataSet 最后一列放入 classList
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0] 
    
    # 如果数据集已经处理了所有属性，但是类标签依然不是唯一的，采用 多数表决的方法决定该叶子节点的分类
    if len(dataSet[0]) == 1:  
        return majorityCnt(classList) 
    
    # 得到最好划分，也就是信息增益最大的列号
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 将 信息增益最大的列的列名存入 bestFeatLabel
    bestFeatLabel = labels[bestFeat]
    # 定义树，存为字典形式
    myTree = {bestFeatLabel:{}}
    # 将信息增益最大的列名删除
    del(labels[bestFeat])
    
    # 将信息增益最大的列取出
    featValues = [example[bestFeat] for example in dataSet]
    # 去除冗余
    uniqueVals = set(featValues)
    # 迭代取值
    for value in uniqueVals:
        # 这行代码复制了类标签
        subLabels = labels[:]  
        # 递归创建树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels) # 字典的嵌套
    # 返回创建好的树
    return myTree

# 使用决策树的分类函数
# inputTree 创建好的决策树
# featLabels 存放feature名的list
# testVec   预测的feature
def classify(inputTree, featLabels, testVec):
    # 取出决策树的key，存为list，并取第一个key
    firstStr = list(inputTree.keys())[0]
    # 取出第一个key所对应的value
    secondDict = inputTree[firstStr]
    # 取出 firstStr 所在的列号
    featIndex = featLabels.index(firstStr)
    # 这段代码为递归找到类别，依次递归向下找
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    myDat, labels = createDataSet()
    print(labels)

    myTree = createTree(myDat, labels)
    print(myTree)

    # 经过 createTree 已经把labels给破坏了，所以现在要从新获取labels
    myDat, labels = createDataSet()
    print(classify(myTree, labels, [1, 0]))
    print(classify(myTree, labels, [1, 1]))