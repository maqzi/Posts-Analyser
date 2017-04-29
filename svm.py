import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random as rd
from sklearn.svm import SVC
from sklearn import metrics
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
import collections

# Split Data Into Train and Test Data
def splitIntoTestAndTrain(dataFrame,splitBy):
    rand = list(range(len(dataFrame)))
    div = int(len(dataFrame) * splitBy)  #e.g splitBy=0.8 for 80% of data
    rd.shuffle(rand)
    train_df_index = rand[:div]
    test_df_index = rand[div:]
    train_df = dataFrame.iloc[train_df_index, :]
    test_df = dataFrame.iloc[test_df_index, :]
    return train_df,test_df;

#Building the SVM Model
def buildSVMModel(filename):
    df = pd.read_csv(filename+'.csv')
    trainDataFrame,testDataFrame=splitIntoTestAndTrain(df,0.8)
    clf = SVC(kernel='linear', probability=True)
    clf = clf.fit(trainDataFrame.drop('Score', 1), trainDataFrame['Score'])
    predict_SVM = clf.predict_proba(trainDataFrame.drop('Score', 1))
    false_positive_SVM, true_positive_SVM, thresholds = metrics.roc_curve(testDataFrame['Score'], predict_SVM[:, 1])
    auc_SVM = metrics.auc(false_positive_SVM, true_positive_SVM)
    return clf,auc_SVM


