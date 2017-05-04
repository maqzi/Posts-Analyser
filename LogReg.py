import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

def PrepData(filename, cutoff, features):
    df = pd.read_csv(filename+'.csv', sep=",")
    Y = (df.Score >= cutoff) # Hardcoded for score
    df = df.ix[:,features]
    df['Y'] = Y
    df = df.dropna()
    return(df)


def xValLog(filename, features, k, cutoff):
    """Cross-Validation Function
    """
    dataset = PrepData(filename, cutoff, features)
    feats = dataset.drop('Y', axis=1)
    outc = dataset.ix[:,'Y']

    # Doing the random grouping
    ind = np.repeat(np.arange(k+1), (len(outc)/(k))+0.5) # The 0.5 is a simple way to round up
    ind = ind[:len(outc)] # Chops off extra indexes
    np.random.shuffle(ind)

    AUC = []

    for f in range(k): 
        logReg = LogisticRegression()
        logReg.fit(feats.ix[ind!=f,:], outc[ind!=f])
        testLRPred = logReg.predict_proba(feats.ix[ind==f,:])[:,1]
        fpr, tpr, thresholds = roc_curve(outc[ind==f], testLRPred)

        AUC.append(auc(fpr,tpr))

    return(sum(AUC)/len(AUC))




if __name__ == "__main__":
    features = ['Flesch_Reading_Ease_Value', 'Coleman_Liau_Index_Value',
                'Dale_Chall_Readability_Score',
                'Polarity', 'Subjectivity',
                'Code_Count', 'Latex_Count','Punc_Count', 'Y'] # Have to put outcome Y in based on above function
    iot = 'iot_posts_with_readibility_measures'
    ai = 'ai_posts_with_readibility_measures'
    stats = 'stats_posts_with_readibility_measures'

    print(xValLog(iot,features, k=10, cutoff=2)) # Just calculated with specific features
    print(xValLog(ai,features, k=10, cutoff=2))
    print(xValLog(stats,features, k=10, cutoff=2))


    ##Testing cutoffs for ai
    #for i in range(10):
    #    print(xValLog(ai,features, k=10, cutoff=i))
    # Doesn't make much of a difference. I think this can be based more on our
    # own subjective definition of what a "high score" is

