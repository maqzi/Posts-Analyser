import re
import string
from collections import Counter

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

    ## Punctuation
    textLen = len(text) #shortcut to avoid storing two texts for the next step
    text = re.sub('[%s]' % re.escape(string.punctuation),'', text)
    punLen = textLen - len(text)

    ## And removing any whitespace
    text = re.sub( '\s+', ' ', text).strip()

    return {'text':text, 'codeLen':codeLen ,'latLen':latLen, 'punLen':punLen}



test = ("\n"
	"<p>I'm doing analysis using Proton-transfer-reaction mass spectrometry techniques. I've collected data over time (5 measurement steps and repeated samplings for each measurement time). \n"
	"For each measure I observe normalized counts-per-second (ncps) associated with different molecula. I also have a qualitative property associated with my measure. </p>\n\n"
	"<p>So my data set is something like:</p>\n\n"
	"<pre><code>Time (express in days, starting from day 0), Sample (numeric), Type (char), Molecula1 (ncps),...., Molecula40 (ncps)\n"
	"</code></pre>\n\n"
	"<p>I need to conduct a statistical analysis with my data set.\n"
	"I'd like to perform Anova to determine witch molecula are statistically relevant. \n"
	"Can I use 1-way Anova? (I could perform one way anova referring to \"Time\", and also to \"Type\") or could I use 2-way anova?\n"
	"After this step I'll get the moleculas that aren't redundant in my data set.</p>\n\n"
    "<p>Suppose we have a polynomial $xy+f(x,y)$, where $f(x,y)$ is a polynomial in $\\mathbb C[x,y]$ whose the lowest degree term has degree at least 3. My question is, are we always able to decompose $xy+f(x,y)$ into the product of two convergent power series $x+g(x,y)$ and $y+h(x,y)$ in a neighborhood of $(0,0)$, where terms in $g$ and $h$ have order higher than 1?</p> \n"
"<p>I have no idea about the convergence of power series with two or more variables. Any solution or reference will be appreciated! Please also note that it's not enough to simply decompose $xy+f(x,y)$ formally into two $x+g(x,y)$ and $y+h(x,y)$. We also need the convergence of $g$ and $h$! </p> \n"
	"<p>Am I right?</p>\n\n"
	"<p>Then I want to use PCA analysis to reduce my independent variables</p>\n\n"
	"<pre><code>Molecula1,...., Molecula40\n"
	"</code></pre>\n\n"
	"<p>Is PCA the right approach or MFA (using FactoMineR ) is better?</p>\n\n"
	"<p>Thanks </p>\n"
	"    </div>\n")

print(TextCleaning("Here's some text with nothing specific"))
outcome = TextCleaning(test)
print(outcome)
