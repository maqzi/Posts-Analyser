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

    istest = np.repeat(np.repeat([False,True],[8,2]), 1+len(outc)/10)[:len(outc)]
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
    istest = np.repeat(np.repeat([False,True],[8,2]), 1+row/10)[:row]
    np.random.shuffle(istest)
    #pdb.set_trace()

    if count: vect = CountVectorizer()
    #else: vect = TfidfVectorizer(ngram_range=(1,3))
    else: vect = TfidfVectorizer()
    fitted = vect.fit_transform(df.ix[~istest,textName])
    test = vect.transform(df.ix[istest,textName])

    logReg = LogisticRegression()
    logReg.fit(fitted, outc[~istest])
    pred = logReg.predict(test)
    probPred = logReg.predict_proba(test)
    
    return(ReportCard(outc[istest],pred,probPred))

def LogRegComb(filename, feature, outcome, textName, count=True):
    """Just gets predictions from tfidf vectorization, then uses that as a
    predictor for the 
    """
    # Initial Dataframe for Vectors
    df = pd.read_csv(filename+'.csv', sep=",")
    df = df.ix[:,[textName]+[outcome]]
    df = df.dropna()

    # Outcome for all data
    outc = df.ix[:,outcome]

    # Training and Test Indices
    row = df.shape[0]
    istest = np.repeat(np.repeat([False,True],[8,2]), 1+row/10)[:row]
    np.random.shuffle(istest)

    # Running Vectorization on Training set
    vect = TfidfVectorizer()
    fitted = vect.fit_transform(df.ix[~istest,textName])

    # Fitting vectorization to log regression and getting prediction
    logReg = LogisticRegression()
    logReg.fit(fitted, outc[~istest])


    # Combining predictions with other features and fitting final reg
    dataset = PrepData(filename, features,outcome)
    feats = dataset.ix[~istest,features] # Feats is JUST TRAINING
    feats['VecPred'] = logReg.predict_proba(fitted)[:,1]
    logReg2 = LogisticRegression()
    logReg2.fit(feats, outc[~istest])

    # Plugging in the test data
    test = vect.transform(df.ix[istest,textName])
    testFeats = dataset.ix[istest,features]
    testFeats['VecPred'] = logReg.predict_proba(test)[:,1]
    preds = logReg2.predict(testFeats)
    probPreds = logReg2.predict_proba(testFeats)
    return(ReportCard(outc[istest],preds,probPreds))

def baseline(filename, binOutcome):
        ## Prints the result statistics for a baseline model
        out = pd.read_csv(filename+'.csv', sep=",")[binOutcome]
        staticAssign = np.repeat(0,len(out)) # Assigning all to the most common value
        if (out.value_counts()[0] < out.value_counts()[1]): staticAssign += 1

        blRep1 = ReportCard(out, staticAssign, np.vstack((1-staticAssign,staticAssign)).T)

        randomAssign = np.random.choice([0,1],len(out)) # Looking at random assignment
        blRep2 = ReportCard(out, randomAssign, np.vstack((1-randomAssign,randomAssign)).T)

        print(blRep1['accuracy'], blRep1['auc'], blRep1['log_loss'])
        print(blRep1['report'])

        ## Decided I don't really need the random assignment, its all just 50%
        #rint(blRep2['accuracy'], blRep2['auc'], blRep2['log_loss'])
        #rint(blRep2['report'])

    
if __name__ == "__main__":
    features = ['Flesch_Reading_Ease_Value', 'Coleman_Liau_Index_Value',
                'Dale_Chall_Readability_Score',
                'Polarity', 'Subjectivity',
                'Code_Count', 'Latex_Count','Punc_Rate'] # Have to put outcome Y in based on above function
    iot = 'iot_posts_with_readibility_measures_score_adj'
    ai = 'ai_posts_with_readibility_measures_score_adj'
    stats = 'stats_posts_with_readibility_measures_score_adj'

    #print(xValLog(iot,features, k=10, cutoff=2)) # Just calculated with specific features
    #print(xValLog(ai,features, k=10, cutoff=2))
    #print(xValLog(stats,features, k=10, cutoff=2))

    ## Testing cutoffs for ai
    ## for i in range(10):
    ##    print(xValLog(ai,features, k=10, cutoff=i))
    ## Doesn't make much of a difference. I think this can be based more on our
    ## own subjective definition of what a "high score" is

    iotRep = LogMod(iot,features,'ScoreLabel', cutoff=2)
    aiRep = LogMod(ai,features,'ScoreLabel', cutoff=2)
    statsRep = LogMod(stats,features,'ScoreLabel', cutoff=2)
    print("Using Features")
    print(iotRep['accuracy'],aiRep['accuracy'],statsRep['accuracy'])
    print(iotRep['auc'],aiRep['auc'],statsRep['auc'])

    ## Vectorized Count
    iotRepVec = LogModVec(iot,'ScoreLabel','Clean_Text',count=True)
    aiRepVec = LogModVec(ai,'ScoreLabel','Clean_Text',count=True)
    statsRepVec = LogModVec(stats,'ScoreLabel','Clean_Text',count=True)

    print("Count Vectorizer")
    print(iotRepVec['accuracy'],aiRepVec['accuracy'],statsRepVec['accuracy'])
    print(iotRepVec['auc'],aiRepVec['auc'],statsRepVec['auc'])

    ## Vectorized TF-IDF
    iotRepVecT = LogModVec(iot,'ScoreLabel','Clean_Text',count=False)
    aiRepVecT = LogModVec(ai,'ScoreLabel','Clean_Text',count=False)
    statsRepVecT = LogModVec(stats,'ScoreLabel','Clean_Text',count=False)

    print("TF-IDF Vectorizer")
    print(iotRepVecT['accuracy'],aiRepVecT['accuracy'],statsRepVecT['accuracy'])
    print(iotRepVecT['auc'],aiRepVecT['auc'],statsRepVecT['auc'])

    iotRepComb = LogRegComb(iot,features,'ScoreLabel','Clean_Text',count=False)
    aiRepComb = LogRegComb(ai,features,'ScoreLabel','Clean_Text',count=False)
    statsRepComb = LogRegComb(stats,features,'ScoreLabel','Clean_Text',count=False)

    print("Two-stage Log Regression")
    print(iotRepComb['accuracy'],aiRepComb['accuracy'],statsRepComb['accuracy'])
    print(iotRepComb['auc'],aiRepComb['auc'],statsRepComb['auc'])


# ## Saved Here Just in Case
# Using Features
# 0.796116504854 0.603773584906 0.67629671516
# 0.808946877912 0.469555035129 0.578797290609
# Count Vectorizer)
# 0.912621359223 0.584905660377 0.683287924742
# 0.899100899101 0.542711864407 0.637123451748
# TF-IDF Vectorizer
# 0.893203883495 0.62893081761 0.702719374904
# 0.828205128205 0.640425531915 0.67918525628
