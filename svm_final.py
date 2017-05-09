import pandas as pd
import numpy as np
import random as rd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score

# convert to numeric, drop nas and other [currently] irrelevant features
def preprocess(filename):
    df = pd.read_csv(filename+'.csv', sep=",")
    # removing unnecessary columns. keeping only numbers atm
    unnecessary = ['Body','ClosedDate','CommunityOwnedDate','CreationDate','Id','LastActivityDate',
              'LastEditDate','LastEditorUserId','LastEditorDisplayName','OwnerDisplayName',
              'OwnerUserId','ParentId','Tags','Title','Clean_Text','AcceptedAnswerId','Score',
              'AnswerCount', 'CommentCount', 'FavoriteCount', 'PostTypeId','ViewCount']
    droppable = np.intersect1d(df.columns,unnecessary)
    df = df.drop(droppable, 1)
    df = df.dropna()
    if df.shape[0]>30000:
        df = df[:30000]
    return df

# Split Data Into Train and Test Data
def splitIntoTestAndTrain(dataFrame,splitBy):
    rand = list(range(len(dataFrame)))
    div = int(len(dataFrame) * splitBy)
    rd.shuffle(rand)
    train_df_index = rand[:div]
    test_df_index = rand[div:]
    train_df = dataFrame.iloc[train_df_index, :]
    test_df = dataFrame.iloc[test_df_index, :]
    return train_df,test_df;

#Best SVM Classifier
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn import svm, grid_search
def svc_param_selection(X, y, nfolds):
    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [10.**i for i in np.arange(-2,2)]},
                    {'kernel': ['linear'], 'C': [10.**i for i in np.arange(-2,2)]}]
    #param_grid = [{'C': [10.**i for i in np.arange(-2,2)], 'kernel': ['linear']}]
    grid_search = GridSearchCV(svm.SVC(probability=True), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    #clf=SVC(kernel='linear', probability=True)
    #clf.fit(X,y)
    #return clf
    return grid_search.best_estimator_


def reportCard(y_true, y_pred, y_proba,name):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    %matplotlib inline
    print('\nReport for:',name)
    target_names = ['low', 'high']
    print(metrics.classification_report(y_true, y_pred, target_names=target_names))
    print("Confusion Matrix\n",metrics.confusion_matrix(y_true,y_pred))
    print("F1 Score:",metrics.f1_score(y_true,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_true,y_pred))
    print("Log Loss:",metrics.log_loss(y_true,y_pred))
    print("AUC Score:",metrics.roc_auc_score(y_true,y_proba[:,1]))
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_proba[:,1])
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for {}'.format(name))
    plt.legend(loc='best')
    plt.show()


def svm_run(filename):
    print("\nRunning for",filename)
    df = preprocess(filename)
    print("dataset size is:",df.shape)
    print("columns:",df.columns)
    train,test=splitIntoTestAndTrain(df,0.8)
    print("\nSVM")
    best_svm = svc_param_selection(train.drop('ScoreLabel', 1),train['ScoreLabel'],5)
    grid_svm_pred = best_svm.predict(test.drop('ScoreLabel', 1))
    grid_svm_pred_prob = best_svm.predict_proba(test.drop('ScoreLabel', 1))
    print("*****BOF: {}*****".format(filename))
    reportCard(test['ScoreLabel'], grid_svm_pred, grid_svm_pred_prob,'GridSearched SVM')
    print("*****EOF: {}*****".format(filename))
if __name__ == "__main__":
	grid_svm_pred1 = clf.predict(test.drop('ScoreLabel', 1))
	grid_svm_pred_prob1 = clf.predict_proba(test.drop('ScoreLabel', 1))
	reportCard(test['ScoreLabel'], grid_svm_pred1, grid_svm_pred_prob1,'GridSearched SVM')