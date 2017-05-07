import pandas as pd
import numpy as np
import pdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from ReportCard import ReportCard, PlotReport
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def PrepData(filename, features, outcome):
    df = pd.read_csv(filename+'.csv', sep=",")
    df = df.ix[:,features+[outcome]]
    df = df.dropna()
    return(df)

def PrepDataVect(filename, features, outcome, textName):
    df = PrepData(filename,features+[textName],outcome)
    vect = CountVectorizer()
    #pdb.set_trace()
    #df['Vect'] = vect.fit_transform(df.ix[:,textName])
    counts = vect.fit_transform(df.ix[:,textName])
    pd.concat([df,pd.DataFrame(counts.A, columns=vect.get_feature_names()).to_string()])
    return(df)

# def xValLog(filename, features, outcome):
#     """Cross-Validation Function
#     """
#     dataset = PrepData(filename, features, outcome)
#     feats = dataset.ix[:,features]
#     outc = dataset.ix[:,outcome]

#     # Doing the random grouping
#     ind = np.repeat(np.arange(k+1), (len(outc)/(k))+0.5) # The 0.5 is a simple way to round up
#     ind = ind[:len(outc)] # Chops off extra indexes
#     np.random.shuffle(ind)

#     AUC = []

#     for f in range(k): 
#         logReg = LogisticRegression()
#         logReg.fit(feats.ix[ind!=f,:], outc[ind!=f])
#         testLRPred = logReg.predict_proba(feats.ix[ind==f,:])[:,1]
#         fpr, tpr, thresholds = roc_curve(outc[ind==f], testLRPred)

#         AUC.append(auc(fpr,tpr))

#     return(sum(AUC)/len(AUC))




def LogMod(filename, features, outcome, cutoff, vect=False):
    """Selects 90% train, 10% test data and fits the logistic regression
    """
    if vect:
        dataset = PrepDataVect(filename, features,outcome,'Clean_Text')
        feats = dataset.drop(outcome)
    else:
        dataset = PrepData(filename, features,outcome)
        feats = dataset.ix[:,features]

    outc = dataset.ix[:,outcome]

    istest = np.repeat(np.repeat([False,True],[9,1]), 1+len(outc)/10)[:len(outc)]
    np.random.shuffle(istest)

    logReg = LogisticRegression()
    logReg.fit(feats.ix[~istest,:], outc[~istest])
    pred = logReg.predict(feats.ix[istest,:])
    probPred = logReg.predict_proba(feats.ix[istest,:])
    return(ReportCard(outc[istest],pred,probPred))


def LogModVec(filename, outcome, textName, count=True):
    """Just Vectorizing, if not count, then tjdblah
    """
    df = pd.read_csv(filename+'.csv', sep=",")
    df = df.ix[:,[textName]+[outcome]]
    df = df.dropna()

    outc = df.ix[:,outcome]

    row = df.shape[0]
    istest = np.repeat(np.repeat([False,True],[9,1]), 1+row/10)[:row]
    np.random.shuffle(istest)
    #pdb.set_trace()

    if count: vect = CountVectorizer()
    else: vect = TfidfVectorizer()
    fitted = vect.fit_transform(df.ix[~istest,textName])
    test = vect.transform(df.ix[istest,textName])

    logReg = LogisticRegression()
    logReg.fit(fitted, outc[~istest])
    pred = logReg.predict(test)
    probPred = logReg.predict_proba(test)
    
    return(ReportCard(outc[istest],pred,probPred))

if __name__ == "__main__":
    features = ['Flesch_Reading_Ease_Value', 'Coleman_Liau_Index_Value',
                'Dale_Chall_Readability_Score',
                'Polarity', 'Subjectivity',
                'Code_Count', 'Latex_Count','Punc_Rate'] # Have to put outcome Y in based on above function
    iot = 'iot_posts_with_readibility_measures'
    ai = 'ai_posts_with_readibility_measures'
    stats = 'stats_posts_with_readibility_measures'

    # print(xValLog(iot,features, k=10, cutoff=2)) # Just calculated with specific features
    # print(xValLog(ai,features, k=10, cutoff=2))
    # print(xValLog(stats,features, k=10, cutoff=2))


    ##Testing cutoffs for ai
    #for i in range(10):
    #    print(xValLog(ai,features, k=10, cutoff=i))
    # Doesn't make much of a difference. I think this can be based more on our
    # own subjective definition of what a "high score" is


    iotRep = LogMod(iot,features,'ScoreLabel', cutoff=2)
    aiRep = LogMod(ai,features,'ScoreLabel', cutoff=2)
    statsRep = LogMod(stats,features,'ScoreLabel', cutoff=2)

    print(iotRep['accuracy'],aiRep['accuracy'],statsRep['accuracy'])
    print(iotRep['auc'],aiRep['auc'],statsRep['auc'])


    ## Vectorized
    iotRepVec = LogModVec(iot,'ScoreLabel','Clean_Text',count=False)
    aiRepVec = LogModVec(ai,'ScoreLabel','Clean_Text',count=False)
    statsRepVec = LogModVec(stats,'ScoreLabel','Clean_Text',count=False)

    print(iotRepVec['accuracy'],aiRepVec['accuracy'],statsRepVec['accuracy'])
    print(iotRepVec['auc'],aiRepVec['auc'],statsRepVec['auc'])

