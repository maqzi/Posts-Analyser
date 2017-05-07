# Date: 04/21/2017
# Updated DFarnand 4/27/2017
from TextCleaning import TextCleaning
import re
import string
from collections import Counter
from textstat.textstat import textstat
import pandas as pd
import numpy as np
def AddReadabilityMeasures(filename):
    df = pd.read_csv(filename+'.csv',index_col=0) #Should be in the same working directory?
    Flesch_Reading_Ease_Value=[]
    Coleman_Liau_Index_Value=[]
    Dale_Chall_Readability_Score=[]
    Code_Count=[]
    Latex_Count=[]
    Text_Len=[]
    #Punc_Count=[]
    Punc_Rate=[]
    Clean_Text=[]
    Polar=[]
    Subj=[]
    for text in df['Body']:
        try:
            cleaned = TextCleaning(text) # Added nan insertion into TextCleaning
        except TypeError:
            print("TypeError (probably reduced to bad text):",text)
            cleaned = TextCleaning('') # Just to get NA values
        Flesch_Reading_Ease_Value.append(cleaned['flesch_reading_ease'])
        Coleman_Liau_Index_Value.append(cleaned['coleman_liau_index'])
        Dale_Chall_Readability_Score.append(cleaned['dale_chall_readability_score'])
        Code_Count.append(cleaned['codeLen'])
        Latex_Count.append(cleaned['latLen'])
        Text_Len.append(cleaned['textLen'])
        #Punc_Count.append(cleaned['punLen'])
        Punc_Rate.append(cleaned['punRate'])
        Clean_Text.append(cleaned['text'])
        Polar.append(cleaned['polarity'])
        Subj.append(cleaned['subjectivity'])
    df['Flesch_Reading_Ease_Value']=Flesch_Reading_Ease_Value
    df['Coleman_Liau_Index_Value']=Coleman_Liau_Index_Value
    df['Dale_Chall_Readability_Score']=Dale_Chall_Readability_Score
    df['Code_Count']=Code_Count
    df['Latex_Count']=Latex_Count
    df['Clean_Text']=Clean_Text
    df['Text_Length']=Text_Len
    #df['Punc_Rate']=(np.array(Punc_Count)/[len(x) for x in Clean_Text]).tolist()
    df['Punc_Rate']=Punc_Rate
    df['Polarity']=Polar
    df['Subjectivity']=Subj
    df['ScoreLabel']= (np.log10(df['Score'])>np.log10(2))*1 #Log because scores skewed
    return df

## Commented to be able to quickly run the script for others
dataFrameAi = AddReadabilityMeasures('ai_posts')
dataFrameStats = AddReadabilityMeasures('stats_posts')
dataFrameIot=AddReadabilityMeasures('iot_posts')
dataFrameAi.to_csv('ai_posts_with_readibility_measures.csv', index=False)
dataFrameIot.to_csv('iot_posts_with_readibility_measures.csv', index=False)
dataFrameStats.to_csv('stats_posts_with_readibility_measures.csv', index=False)


