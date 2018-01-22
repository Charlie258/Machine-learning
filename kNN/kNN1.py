import numpy as np
import operator
import pandas as pd
import csv

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
    train = pd.read_csv(filename)
    returnVect = train.values

    returnMat_all = returnVect[:, 1:]
    classLabelVector_all = returnVect[:, 0]

    # returnMat = returnMat_all[0:42000, :]
    # classLabelVector = classLabelVector_all[0:42000]
    return returnMat_all, classLabelVector_all


def datingClassTest():
    # 取出10%的数据作为测试样例
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('kNN_data/train.csv')
    m = datingDataMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):

        # 输入参数:normMat[i,:]为测试样例，表示归一化后的第i行数据
        #       normMat[numTestVecs:m,:]为训练样本数据，样本数量为(m-numTestVecs)个
        #       datingLabels[numTestVecs:m]为训练样本对应的类型标签
        #       k为k-近邻的取值
        classifierResult = classify0(datingDataMat[i, :], datingDataMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

def img2vector(filename):
    train = pd.read_csv(filename)
    returnVect = train.values
    return returnVect


def saveResult_problem(result):
    with open('result.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)

def saveResult(result):
    df = pd.DataFrame(result)
    df.index += 1
    df.index.name = 'ImageId'
    df.columns = ['Label']
    df.to_csv('results.csv', header=True)

def handwritingClassTest():
    ## step 1: Getting training set
    datingDataMat, datingLabels = file2matrix('kNN_data/train.csv') # load the training set
    m = datingDataMat.shape[0]

    ## step 2: Getting testing set
    testDataMat = img2vector('kNN_data/test.csv')  # load the testing set
    labelfile = img2vector('kNN_data/knn_benchmark.csv')
    testDataLabel = labelfile[:, 1]
    numTestData = int(testDataMat.shape[0])
    errorCount = 0.0
    resultList = []
    for i in range(numTestData):
        classifierResult = classify0(testDataMat[i, :], datingDataMat, datingLabels, 3)
        resultList.append(classifierResult)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, testDataLabel[i]))
        if (classifierResult != testDataLabel[i]):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(numTestData)))
    saveResult(resultList)

handwritingClassTest()
#datingClassTest()
# vectorUnderTest = img2vector('kNN_data/sample_submission.csv')
# hwLabels = vectorUnderTest[:, 1:]
# print(hwLabels)


