##############################################################################
# Daniel Farnand                                                             #
# 12 April 2017                                                              #
# Code to import the Stackexchange data from archive.org                     #
##############################################################################

# Observations:
# 1. Posts.xml
#    - Each row is a post


import xml.etree.ElementTree as etree

posts = etree.parse("ai.stackexchange/Posts.xml").getroot()

attribs = list()

for post in posts:
    attribs.append(post.attrib)


###########################################
# Examples of how to pull data from this: #
###########################################


# The list of all attributes we can pick from
attribs[1].keys()

# Manually pulling specific values.
attribs[1].get('Id')

# Getting a list of all post text
postText = list()
for post in attribs:
    postText.append(post.get('Body'))
# This just a list of strings - note that this contains html formatting. It should be useful, especially for separating code out from prose.
print(postText[13:16])

# Getting list of scores
Scores = list()
for post in attribs:
    Scores.append(post.get('Score'))

#################################################
# Seeing about matching with Comments and Votes #
#################################################

comments = etree.parse("ai.stackexchange/Comments.xml").getroot()
votes = etree.parse("ai.stackexchange/Votes.xml").getroot()

commAtt = list()
for c in comments:
    commAtt.append(c.attrib)

voteAtt = list()
for v in votes:
    voteAtt.append(v.attrib)


i = 1
attribs[i]
commAtt[i]
voteAtt[i]

# Observations
# - We can use 'PostID' to connect sets from different data.
# - I believe this corresponds with 'Id' in Posts


# Looking at the above comment, we have a comment to postid 7. In order to
# retrieve this post:

for p in attribs:
    if p.get('Id') == '7': 
        print(p)

# Goal - more efficient way to do this. Probably involves putting the info all
# together into a data frame.
