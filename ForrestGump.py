# RANDOM FOREST
# @Author: Munaf
# @Date: 04/27/17

# Import necessary stuff
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# convert to numeric, drop nas and other [currently] irrelevant features
def preprocess(filename):
    df = pd.read_csv(filename+'.csv', sep=",")
    df.fillna(np.NaN, inplace=True)

    # removing unnecessary columns. keeping only numbers atm
    unnecessary = ['Body','ClosedDate','CommunityOwnedDate','CreationDate','Id','LastActivityDate',
              'LastEditDate','LastEditorUserId','LastEditorDisplayName','OwnerDisplayName','OwnerUserId','ParentId',
              'Tags','Title','Clean_Text','AcceptedAnswerId']
    droppable = np.intersect1d(df.columns,unnecessary)
    df = df.drop(droppable, 1)
    for i in df.drop('ScoreLabel',1).columns:
        df[i] = df[i].replace([np.NaN], df[i].mean(skipna=True, axis=0))
        # df[i] = StandardScaler().fit_transform(df[i], df['ScoreLabel']) #Should scale? doesn't affect the result
    df = df.apply(pd.to_numeric)
    return df

# Getting the best random forest parameters
def gridSearching(rfc, X, Y):
    '''
    All Parameters in a RFC
                'max_depth':[None],
                'min_samples_split':[2],
                'min_samples_leaf':[1],
                'min_weight_fraction_leaf':[0.0],
                'max_leaf_nodes':[None],
                'min_impurity_split':[1e-07],
                'bootstrap':[True],
                'oob_score':[False],
                'n_jobs':[1],
                'random_state':[None],
                'verbose':[0],
                'warm_start':[False],
                'class_weight':[None],
                'max_features: [sqrt],
                'criterion': ['gini'],
                'n_estimators' : [10]
    '''

    param_grid = {'n_estimators': np.arange(10, 100, 10),
              'max_features': ['sqrt', 'log2', None],
              'criterion': ['gini', 'entropy'],
              }

    # Performing a Grid Search to find best RFC
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10)
    CV_rfc.fit(X, Y)
    print("Best parameters were {} with an accuracy of: {}".format(CV_rfc.best_params_,CV_rfc.best_score_))

# Show boosted RFC results
def adaboostedRFC(best_rfc,X,Y):
    from sklearn.ensemble import AdaBoostClassifier

    brfc = AdaBoostClassifier(best_rfc,
                          algorithm="SAMME.R",
                          n_estimators=10)
    bScores = cross_val_score(brfc, X, Y, cv=10)
    print('crossvalidated accuracy after Adaboost: {}'.format(bScores.mean()))

def RunForrestRun(filename):
    print("\nRunning for",filename)
    df = preprocess(filename)
    print("dataset size is:",df.shape)

    X = df.drop('ScoreLabel', 1)
    Y = df['ScoreLabel']

    # gridSearching(RandomForestClassifier(),X,Y)

    ## RUN after first calculating the best parameters using the gridSearching function.
    ## PS. USE THE BEST ONES IN THE CLASSIFIER'S ARGUMENTS
    best_rfc = RandomForestClassifier(max_features='log2', n_estimators=10, criterion='gini')
    scores = cross_val_score(best_rfc, X, Y, cv=10)
    print('crossvalidated accuracy: {}'.format(scores.mean()))

    adaboostedRFC(best_rfc,X,Y)

if __name__ == "__main__":
    iot = 'iot_posts_with_readibility_measures'
    ai = 'ai_posts_with_readibility_measures'
    RunForrestRun(iot)
    RunForrestRun(ai)