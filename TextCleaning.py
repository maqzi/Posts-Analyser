import re
import string
from collections import Counter
from textstat.textstat import textstat

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
    fre = textstat.flesch_reading_ease(text)
    cl = textstat.coleman_liau_index(text)
    dc = textstat.dale_chall_readability_score(text)

    ## Punctuation
    textLen = len(text) #shortcut to avoid storing two texts for the next step
    text = re.sub('[%s]' % re.escape(string.punctuation),'', text)
    punLen = textLen - len(text)

    ## And removing any whitespace
    text = re.sub( '\s+', ' ', text).strip()

    return {'text':text, 'codeLen':codeLen ,'latLen':latLen, 'punLen':punLen,
            'flesch_reading_ease':fre, 'coleman_liau_index':cl,
            'dale_chall_readability_score':dc }

