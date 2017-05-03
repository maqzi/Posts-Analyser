import re
import string
import numpy as np
from collections import Counter
from textstat.textstat import textstat
from textblob import TextBlob


def TextCleaning(text):
    """Function to clean data from Stackexchange dumps

    - Takes the html as a string
    - Removes newlines
    - If code blocks exist, records length of codeblocks (otherwise set to 0).
    - Then removes code
    - Removes html tags
    - Counts and removes for latex
    - returns the text and the two counts (more can be added)
    """

    NullDict = {'text':np.nan, 'codeLen':np.nan ,'latLen':np.nan, 'punLen':np.nan,
                'flesch_reading_ease':np.nan, 'coleman_liau_index':np.nan,
                'dale_chall_readability_score':np.nan,
                'polarity': np.nan, 'subjectivity': np.nan} # To return when we want all nulls

    if not isinstance(text, str): return NullDict # Catches non-text

    ## Removes line breaks and carriage returns (latter prob unnecessary)
    text = text.replace('\n', ' ').replace('\r', '') 

    ## Code Blocks
    # combines code blocks all together to count characters
    codeLen = len(' '.join(re.findall(r"<code>(.*?)</\code>", text, re.DOTALL)))
    text = re.sub(r"<code>(.*?)</\code>",'',text) ## Removes code

    ## Removing HTML
    text = re.sub(re.compile('<.*?>'), '', text)

    ## LaTeX
    latLen = len(' '.join(re.findall(r"\$(.*?)\$", text, re.DOTALL)))
    text = re.sub(r"(\$.*?\$)",'',text) ## Removes code

    ## Calculate Readibility Scores
    if len(text)-text.count(' ') > 0: # Don't want to feed just whitespace in
        fre = textstat.flesch_reading_ease(text)
        cl = textstat.coleman_liau_index(text)
        dc = textstat.dale_chall_readability_score(text)
        sent = TextBlob(text).sentiment
    else:
        return NullDict

    ## Punctuation
    textLen = len(text) #shortcut to avoid storing two texts for the next step
    text = re.sub('[%s]' % re.escape(string.punctuation),'', text)
    punLen = textLen - len(text)

    ## And removing any whitespace
    text = re.sub( '\s+', ' ', text).strip()

    return {'text':text, 'codeLen':codeLen ,'latLen':latLen, 'punLen':punLen,
            'flesch_reading_ease':fre, 'coleman_liau_index':cl,
            'dale_chall_readability_score':dc,
            'polarity': sent.polarity, 'subjectivity': sent.subjectivity}

