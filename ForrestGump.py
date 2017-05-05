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

    # df = AppendNaiveBayesGuess(df,filename)
    # df.fillna(np.NaN, inplace=True)

    # removing unnecessary columns. keeping only numbers atm
    unnecessary = ['Body','ClosedDate','CommunityOwnedDate','CreationDate','Id','LastActivityDate',
              'LastEditDate','LastEditorUserId','LastEditorDisplayName','OwnerDisplayName','OwnerUserId','ParentId',
              'Tags','Title','Clean_Text','AcceptedAnswerId','Score']

    droppable = np.intersect1d(df.columns,unnecessary)
    df = df.drop(droppable, 1)
    for i in df.drop(['ScoreLabel','nb_guess'],1).columns:
        df[i] = df[i].replace([np.NaN], df[i].mean(skipna=True, axis=0))
        df[i] = StandardScaler().fit_transform(df[i], df['ScoreLabel']) #zero mean + unit variance
    # df = df.apply(pd.to_numeric)
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
    return CV_rfc.best_estimator_

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

    best_rfc =  gridSearching(RandomForestClassifier(),X,Y)
    print('score over complete set: {}'.format(best_rfc.score()))

    # adaboostedRFC(best_rfc,X,Y)
    XGBoosting(X,Y,filename)

def AppendNaiveBayesGuess(df,filename):
    df['nb_guess'] = pd.read_csv('nb_'+filename+'_guess.csv', header=None,index_col=0)
    return df

def XGBoosting(X,y,filename):
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

    # ROW SAMPLING
    # grid search
    model = XGBClassifier()
    subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    param_grid = dict(subsample=subsample)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, y)
    print("accuracy for row sampling:", grid_result.best_estimator_.score(X, y))

    # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))


    # COLUMN SAMPLING
    # grid search
    model = XGBClassifier()
    colsample_bytree = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    param_grid = dict(colsample_bytree=colsample_bytree)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, y)

    print("accuracy for col sampling:", grid_result.best_estimator_.score(X, y))

    # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    # SPLIT SAMPLE
    # grid search
    model = XGBClassifier()
    colsample_bylevel = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    param_grid = dict(colsample_bylevel=colsample_bylevel)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, y)

    print("accuracy for split sampling:", grid_result.best_estimator_.score(X, y))

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



if __name__ == "__main__":
    iot = 'iot_posts_with_readibility_measures'
    ai = 'ai_posts_with_readibility_measures'
    stats = 'stats_posts_with_readibility_measures'
    RunForrestRun(iot)
    RunForrestRun(ai)
    RunForrestRun(stats)