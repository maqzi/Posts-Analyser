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
    Punc_Count=[]
    Clean_Text=[]
    for text in df['Body']:
        if(isinstance(text, str)):
            cleaned = TextCleaning(text)
            text = cleaned['text']
            flesch_reading_ease = cleaned['flesch_reading_ease']
            coleman_liau_index = cleaned['coleman_liau_index']
            dale_chall_readability_score = cleaned['dale_chall_readability_score']
            codeLen = cleaned['codeLen']
            latLen = cleaned['latLen']
            punLen = cleaned['punLen']
        else: # Just fills in NA values - these will be removed for analysis
            cleaned = np.nan
            text = np.nan
            flesch_reading_ease = np.nan
            coleman_liau_index = np.nan
            dale_chall_readability_score = np.nan
            codeLen = np.nan
            latLen = np.nan
            punLen = np.nan
        Flesch_Reading_Ease_Value.append(flesch_reading_ease)
        Coleman_Liau_Index_Value.append(coleman_liau_index)
        Dale_Chall_Readability_Score.append(dale_chall_readability_score)
        Code_Count.append(codeLen)
        Latex_Count.append(latLen)
        Punc_Count.append(punLen)
        Clean_Text.append(text)
    df['Flesch_Reading_Ease_Value']=Flesch_Reading_Ease_Value
    df['Coleman_Liau_Index_Value']=Coleman_Liau_Index_Value
    df['Dale_Chall_Readability_Score']=Dale_Chall_Readability_Score
    df['Code_Count']=Code_Count
    df['Latex_Count']=Latex_Count
    df['Punc_Count']=Punc_Count
    df['Clean_Text']=Clean_Text
    return df
dataFrameAi = AddReadabilityMeasures('ai_posts')
dataFrameIot=AddReadabilityMeasures('iot_posts')
dataFrameAi.to_csv('ai_posts_with_readibility_measures.csv', index=False)
dataFrameIot.to_csv('iot_posts_with_readibility_measures.csv', index=False)


