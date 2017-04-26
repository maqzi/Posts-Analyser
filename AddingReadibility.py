# Date: 04/21/2017
# Updated DFarnand 4/27/2017
from TextCleaning import TextCleaning
from textstat.textstat import textstat
import pandas as pd
def AddReadabilityMeasures(filename):
    df = pd.read_csv(filename+'.csv') #Should be in the same working directory?
    Flesch_Reading_Ease_Value=[]
    Coleman_Liau_Index_Value=[]
    Dale_Chall_Readability_Score=[]
    Code_Count=[]
    Latex_Count=[]
    Punc_Count=[]
    for text in df['Body']:
        if(isinstance(text, str)):
            cleaned = TextCleaning(text)
            text = cleaned['text']
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            coleman_liau_index = textstat.coleman_liau_index(text)
            dale_chall_readability_score = textstat.dale_chall_readability_score(text)
        else: #Consider using some sort of NA value? We probably just want to remove these in the analysis anyway
            cleaned = TextCleaning('') # Hacky way have values below
            flesch_reading_ease = 0
            coleman_liau_index =0
            dale_chall_readability_score =0
        Flesch_Reading_Ease_Value.append(flesch_reading_ease)
        Coleman_Liau_Index_Value.append(coleman_liau_index)
        Dale_Chall_Readability_Score.append(dale_chall_readability_score)
        Code_Count.append(cleaned['codeLen'])
        Latex_Count.append(cleaned['latLen'])
        Punc_Count.append(cleaned['punLen'])
    df['Flesch_Reading_Ease_Value']=Flesch_Reading_Ease_Value
    df['Coleman_Liau_Index_Value']=Coleman_Liau_Index_Value
    df['Dale_Chall_Readability_Score']=Dale_Chall_Readability_Score
    df['Code_Count']=Code_Count
    df['Latex_Count']=Latex_Count
    df['Punc_Count']=Punc_Count
    return df
dataFrameAi = AddReadabilityMeasures('ai_posts')
dataFrameIot=AddReadabilityMeasures('iot_posts')
dataFrameAi.to_csv('ai_posts_with_readibility_measures.csv')
dataFrameIot.to_csv('iot_posts_with_readibility_measures.csv')


