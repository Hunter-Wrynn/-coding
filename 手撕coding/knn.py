 #encoding=utf8
import numpy as np
from collections import Counter
class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None
        self.train_vars = 0
    def fit(self, feature, label):
        '''
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        '''
        #********* Begin *********#
        self.train_feature = feature
        self.train_label = label
        self.train_vars = feature.shape[0]
        #********* End *********#
    def predict(self, feature):
        '''
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        '''
        #********* Begin *********#
        # distance = self.calculateDistance(feature)
        # KLabels = self.getKLabels(distance)
        # return self.getAppearMostLabel(KLabels)
        result = []
        for data in feature:
            distance = self.calculateDistance(data)
            KLabels = self.getKLabels(distance)
            result.append(self.getAppearMostLabel(KLabels))
        return result
        #********* End *********#
    def calculateDistance(self, feature):
        diffMat = np.tile(feature, (self.train_vars, 1)) - self.train_feature
        sqDistance = (diffMat ** 2).sum(axis=1)
        return sqDistance ** 0.5
    def getKLabels(self, distance):
        argOder = distance.argsort()[0:self.k]
        return (self.train_label[i] for i in argOder)
    def getAppearMostLabel(self, KLabels):
        label, count = Counter(KLabels).most_common(1)[0]
        return label