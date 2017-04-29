import pandas as pd
import numpy as np

def remove_punct(desc):
    import string
    import re
    t = re.sub('[' + string.punctuation + ']', '', str(desc).lower())
    return t

def NaiveBayesGuess(df,filename):
    from textblob.classifiers import NaiveBayesClassifier
    import pickle
    texts = df['Clean_Text']
    scores = df['ScoreLabel']
    text_tuples = []

    # Create Text Tuples
    for i,text in enumerate(texts):
        text = remove_punct(text)
        text_tuples.append(tuple((text,scores[i])))

    # Train a Naive Bayes Classifier (NLTK implementation) on normalized texts
    nbcl = NaiveBayesClassifier(text_tuples)

    # 'Pickle' the classifier for future use
    f = open('nbcl_'+filename+'.pickle', 'wb')
    pickle.dump(nbcl, f)
    f.close()

    # Load Pickled Classifier
    # loaded_nbcl = pickle.load(open('nbcl_'+filename+'.pickle', 'rb'))

    loaded_nbcl = nbcl #comment out when loading

    # Check the classifier's predictions for the test set
    nbcl_acc = loaded_nbcl.accuracy(text_tuples)
    print('NB description classifier: {}'.format(nbcl_acc))

    # Classify all descriptions and write class values to file
    pd.DataFrame(loaded_nbcl.classify(remove_punct(text)) for text in texts).to_csv('nb_'+filename+'_guess.csv',index='False',header='False')


iot = 'iot_posts_with_readibility_measures'
ai = 'ai_posts_with_readibility_measures'
df_iot = pd.read_csv(iot+'.csv', sep=",")
df_ai = pd.read_csv(ai+'.csv', sep=",")
# NaiveBayesGuess(df_iot,iot)
NaiveBayesGuess(df_ai,ai)