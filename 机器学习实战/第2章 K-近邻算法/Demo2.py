
# 导入库
import numpy as np
import matplotlib.pyplot as plt

# 读取文本数据
def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取
    arrayOLines = fr.readlines()
    # 得到文件共计多少行
    numberOfLines = len(arrayOLines)
    # 定义 feature 容器
    returnMat = np.ones((numberOfLines, 3))
    # 定义 label 容器
    classLabelVector = []
    # 技术索引
    index = 0
    # 一次读取一行，共计读取 numberOfLines 行
    for line in arrayOLines:
        # line.strip() 截取掉所有回车字符(即每行最后一个字符)，再每行以'\t' 分裂成一个列表
        line = line.strip().split('\t')
        # 每行前三个元素，依次放入 feature 容器中
        returnMat[index, :] = line[0:3]
        # label 放容器中
        classLabelVector.append(int(line[-1]))
        # 索引 +1
        index += 1
    
    # 返回 feature 和 label
    return returnMat, classLabelVector


#准备数据：归一化数值
def autoNorm(dataSet):
    # shape = (3,) 得到每一列的最小值
    minVals = dataSet.min(0)
    # shape = (3,) 得到每一列的最大值
    maxVals = dataSet.max(0)
    # shape = (3,) 每一列的最大值 - 最小值
    ranges = maxVals - minVals
    # 定义存放归一化数据的容器，shape 和 归一化之前的shape相同
    normDataSet = np.zeros(np.shape(dataSet))
    # 行数
    m = dataSet.shape[0]
    # 每个元素减去该列最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 归一化公式 newValue = (oldValue - min) / (max - min)
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化之后的 feature，
    return normDataSet, ranges, minVals


# k-近邻算法
def classify0(x, dataSet, labels, k):
    # size 为样本数，即 dataSet 行数
    size = dataSet.shape[0]
    # 把 x 整理成和 dataSet 一样的shape，然后对应元素相减
    diffMat = np.tile(x, (size, 1)) - dataSet
    # 每一个元素取平方
    sqDiffMat = diffMat ** 2
    # 行相加
    sqDistances = sqDiffMat.sum(axis = 1)
    # 现在shape为 (4,)， 每行开根号
    distances = sqDistances ** 0.5
    # 返回元素由小到大排序的下标
    sortedDisIndicies = distances.argsort()
    # 定义字典，存放计数
    classCount = {}
    # 循环 k 次，即按照距离，取出前 k 个
    for i in np.arange(k):
        # 按照从小到大，依次取出相应的 label
        voteIlabel = labels[sortedDisIndicies[i]]
        # 如果字典中没该 label 就添加 value 为 1 ，如果有，则在原来基础上 value + 1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    # 找出字典中value最大的label，也就是前 k 个中出现次数最多的 label
    value = -1
    for key in classCount:
        if value < classCount[key]:
            flag = key
            value = classCount[key]
            
    return flag

    
# 分类器针对约会网站的测试代码
# 用已有数据的 90% 作为训练样本，10% 数据去测试分类器
def datingClassTest():
    hoRatio = 0.10 # 比率
    # 调用 file2matrix 函数，得到文本中的 feature 和 label
    datingDataMat, datingLabels = file2matrix(r'E:\tensorflow\datingTestSet2.txt')
    # 准备数据：归一化数值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 得到一共多少个数据
    m = normMat.shape[0]
    # 按比率分开，前 100 个测试，验证分类器准确度，后 900 个作为训练集
    numTestVecs = int(m * hoRatio) # 100
    # 分类错误计数
    errorCount = 0.0
    # 依次取前 100 个数据，作为测试
    for i in np.arange(numTestVecs):
        # 调用 k-近邻算法， 返回 label， k = 3
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # 输出 分类 label 和 正确的 label，观察是否一致
        print("thr classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        # 查看预测 label 和正确 label 是否一致，不一致则错误计数加1
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    # 输出错误率
    print("the total error rate is: %f" %( errorCount / float(numTestVecs)))


if __name__ == '__main__':
    datingClassTest()