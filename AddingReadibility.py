# Date: 04/21/2017
from textstat.textstat import textstat
import pandas as pd
def AddReadabilityMeasures(filename):
    df = pd.read_csv('/Users/amanmahato/Desktop/Local Folder/Posts-Analyser/'+filename+'.csv')
    Flesch_Reading_Ease_Value=[]
    Coleman_Liau_Index_Value=[]
    Dale_Chall_Readability_Score=[]
    for text in df['Body']:
        if(isinstance(text, str)):
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            coleman_liau_index = textstat.coleman_liau_index(text)
            dale_chall_readability_score = textstat.dale_chall_readability_score(text)
        else:
            flesch_reading_ease = 0
            coleman_liau_index =0
            dale_chall_readability_score =0
        Flesch_Reading_Ease_Value.append(flesch_reading_ease)
        Coleman_Liau_Index_Value.append(coleman_liau_index)
        Dale_Chall_Readability_Score.append(dale_chall_readability_score)
    df['Flesch_Reading_Ease_Value']=Flesch_Reading_Ease_Value
    df['Coleman_Liau_Index_Value']=Coleman_Liau_Index_Value
    df['Dale_Chall_Readability_Score']=Dale_Chall_Readability_Score
    return df
dataFrameAi = AddReadabilityMeasures('ai_posts')
dataFrameIot=AddReadabilityMeasures('iot_posts')
dataFrameAi.to_csv('ai_posts_with_readibility_measures.csv')
dataFrameIot.to_csv('iot_posts_with_readibility_measures.csv')


