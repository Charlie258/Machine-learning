import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
	group = np.array([[1.0, 1.1], [1.0, 1.0],[0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy dataSetSize rows for dataSet
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndicies = distances.argsort()

    classCount = {}  # define a dictionary (can be append element)

    for i in range(k):
        ## step 3: choose the min k distance
        voteIlabel = labels[sortedDistIndicies[i]]

        ## step 4: count the times labels occur
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    ## step 5: the max voted class will return
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)

    # get the number of lines
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)

    # set a '0' matrix for storing 3 columns
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()  # 去掉前后多余的回车字符
        listFromLine = line.split('\t')  # 用\t将整行分割成列表
        returnMat[index, :] = listFromLine[0:3]  # 取前三个元素
        classLabelVector.append(int((listFromLine[-1])))  # 存储最后一列
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 取每列的最小值
    maxVals = dataSet.max(0)  # 取每列的最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # tile(A, reps): Construct an array by repeating A reps times
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    # 取出10%的数据作为测试样例
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):

        # 输入参数:normMat[i,:]为测试样例，表示归一化后的第i行数据
        #       normMat[numTestVecs:m,:]为训练样本数据，样本数量为(m-numTestVecs)个
        #       datingLabels[numTestVecs:m]为训练样本对应的类型标签
        #       k为k-近邻的取值
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = {'didntLike': 'not at all', 'smallDoses': 'in small doses', 'largeDoses': 'in large doses'}
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult])


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    ## step 1: Getting training set
    hwLabels = []  # 存储标签
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # get label from file name such as "1_18"
        classNumStr = int(fileStr.split('_')[0])  # return 1
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    ## step 2: Getting testing set
    testFileList = listdir('testDigits')  # load the testing set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


handwritingClassTest()

#classifyPerson()
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#            15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
# plt.show()