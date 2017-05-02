# Author: Aman Mahato
# Date: 04/27/2017

import pandas as pd
import numpy as np
import random as rd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score

# convert to numeric, drop nas and other [currently] irrelevant features
# From ForestGump.py
def preprocess(filename):
    df = pd.read_csv(filename+'.csv', sep=",")
    df.fillna(np.NaN, inplace=True)
    # removing unnecessary columns. keeping only numbers atm
    unnecessary = ['Body','ClosedDate','CommunityOwnedDate','CreationDate','Id','LastActivityDate',
              'LastEditDate','LastEditorUserId','LastEditorDisplayName','OwnerDisplayName','OwnerUserId','ParentId',
              'Tags','Title','Clean_Text','Score','AcceptedAnswerId']
    droppable = np.intersect1d(df.columns,unnecessary)
    df = df.drop(droppable, 1)
    for i in df.drop('ScoreLabel',1).columns:
        df[i] = df[i].replace([np.NaN], df[i].mean(skipna=True, axis=0))
    df = df.apply(pd.to_numeric)
    return df

# Split Data Into Train and Test Data
def splitIntoTestAndTrain(dataFrame,splitBy):
    rand = list(range(len(dataFrame)))
    div = int(len(dataFrame) * splitBy)  #splitBy=0.8 for 80% of data
    rd.shuffle(rand)
    train_df_index = rand[:div]
    test_df_index = rand[div:]
    train_df = dataFrame.iloc[train_df_index, :]
    test_df = dataFrame.iloc[test_df_index, :]
    return train_df,test_df;

#Building the SVM Model
def buildSVMModel(filename):
    df=preprocess(filename)
    trainDataFrame,testDataFrame=splitIntoTestAndTrain(df,0.8)
    clf = SVC(kernel='linear', probability=True)
    clf = clf.fit(trainDataFrame.drop('ScoreLabel', 1), trainDataFrame['ScoreLabel'])
    predict_SVM = clf.predict_proba(testDataFrame.drop('ScoreLabel', 1))
    # import pickle
    # f = open('svm_aman.pickle', 'wb')
    # pickle.dump(clf, f)
    # f.close()
    # loaded_clf = pickle.load(open('nbcl_aman.pickle', 'rb'))
    #false_positive_SVM, true_positive_SVM, thresholds = metrics.roc_curve(testDataFrame['ScoreLabel'], predict_SVM[:, 1])
    #auc_SVM = metrics.auc(false_positive_SVM, true_positive_SVM)
    auc_SVM=metrics.accuracy_score(testDataFrame['ScoreLabel'], clf.predict(testDataFrame.drop('ScoreLabel',1)))
    #from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    crossvalidated_scores = cross_val_score(clf, df.drop('ScoreLabel',1), df['ScoreLabel'], cv=10)
    return clf,crossvalidated_scores.mean(),auc_SVM.mean()

if __name__ == "__main__":
    ai_classifier,score_ai,auc_score_ai=buildSVMModel('ai_posts_with_readibility_measures')
    iot_classifier, score_iot,auc_score_iot=buildSVMModel('iot_posts_with_readibility_measures')
    print('crossvalidated_accuracy_ai: {}'.format(score_ai))
    print('crossvalidated_accuracy_iot: {}'.format(score_iot))
    print('auc_accuracy_ai: {}'.format(auc_score_ai))
    print('auc_accuracy_iot: {}'.format(auc_score_iot))
