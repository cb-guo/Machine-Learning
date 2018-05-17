
# 导入库
import numpy as np
# 从 os 模块中导入函数 listdir ，它可以列出给定目录的文件名
from os import listdir 


# 准备数据：将图像转换为测试向量
# 该函数创建 1x1024 的 NumPy 数组，然后打开给定的文件，循环读出文件的前 32 行，
# 并将每行的前 32 个字符值存储在 NumPy 数组中，最后返回数组
def img2vector(filename):
    # 创建 1x1024 的 NumPy 数组
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 读取 32 行(每个文件中都为32行)
    for i in np.arange(32):
        # 读取一行
        lineStr = fr.readline()
        # 每一行有 32 个数字，循环读取
        for j in np.arange(32):
            # 将每一行读取的字符转化为 int 类型，然后存放到 returnVect 中
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回 1x1024 的 NumPy 数组，即把 32x32 的图片压缩成 1x1024
    return returnVect


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


def handwritingClassTest():
    # label 容器
    hwLabels = []
    # 得到 E:\tensorflow\trainingDigits 文件夹下的文件名
    trainingFileList = listdir(r'E:\tensorflow\trainingDigits')
    # 计算共计多少文件
    m = len(trainingFileList)
    # 定义存放 feature 的容器
    trainingMat = np.zeros((m, 1024))
    # 将每个文件中的 32x32 矩阵取出，放到 trainingMat 中的一行
    for i in np.arange(m):
        # 取出一个文件名， 例如 0_11.txt
        fileNameStr = trainingFileList[i]
        # 在 '.' 分开，取到 0_11
        fileStr = fileNameStr.split('.')[0]
        # 从 '_' 分开，取到 0
        classNumStr = int(fileStr.split('_')[0])
        # 0 即为该图片，32x32 手写数字的 label，放到 hwLabels 中
        hwLabels.append(classNumStr)
        # 调用 img2vector 函数，将该文件中的 32x32 压缩为 1x1024 存放到 trainingMat 的某行中
        trainingMat[i, :] = img2vector(r'E:\tensorflow\trainingDigits\%s' % fileNameStr )
    
    # 得到 E:\tensorflow\testDigits 文件夹下的文件名
    testFileList = listdir(r'E:\tensorflow\testDigits')
    # 分类错误图片计数
    errorCount = 0.0
    # 测试图片总数量
    mTest = len(testFileList)
    # 循环读取文件 (即取出每一张图片)
    for i in np.arange(mTest):
        # 取出一个文件名， 例如 6_12.txt
        fileNameStr = testFileList[i]
        # 在 '.' 分开，取到 6_12
        fileStr = fileNameStr.split('.')[0]
        # 从 '_' 分开，取到 6
        classNumStr = int(fileStr.split('_')[0])
        # 调用 img2vector 函数，将该文件中的 32x32 压缩为 1x1024 存放到 vectorUnderTest
        vectorUnderTest = img2vector(r'E:\tensorflow\testDigits\%s' % fileNameStr)
        # # 调用 k-近邻算法， 返回 label， k = 3
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # 输出 分类 label 和 正确的 label，观察是否一致
        print("the classifier came back with: %d, the real answer is: %d " %(classifierResult, classNumStr))
        # 查看预测 label 和正确 label 是否一致，不一致则错误计数加1
        if classifierResult != classNumStr :
            errorCount += 1.0
    # 打印输出 错误计数总数
    print("\nthe total number of error is: %d" % errorCount)
    # 打印输出 错误率
    print("\nthe total error rate is: %f " %(errorCount / float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()