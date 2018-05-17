
# 导入库
import numpy as np
import matplotlib.pyplot as plt

# 创建数据集
def createDataSet():
    # 创建一个 (4, 2) 的特征
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 定义标签，group中每一行对应一个标签
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 画出数据集
def Figure():
    group, labels = createDataSet()
    for i in range(group.shape[0]):
        # 每次画出一个点
        plt.scatter(group[i,0], group[i,1])
        # 在画出的点附近画出文字
        plt.text(group[i,0]-0.04, group[i,1]-0.01, labels[i])
    plt.show()

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

if __name__ == '__main__':
    group, labels = createDataSet()
    flag = classify0([0, 0], group, labels, 3)
    print(flag)    #  B
    flag = classify0([1, 1], group, labels, 3)
    print(flag)    #  A
    flag = classify0([0.8, 0.8], group, labels, 3)
    print(flag)    #  A
    Figure()