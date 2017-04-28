# Posts-Analyser
Post analysis (and possibly prediction) for questions/answers from Stack Overflow.


## Plan for Analysis of Posts

1. Get data into a Pandas Dataframe

2. Use the following to use as features

- Word count
- Readability index _(find the specifics)_
- Contains Source Code
- Contains Latex math
- Sentiment Analysis (Possibly predictive, but more intended for  descriptive)

3. Run a Predictive Analysis

4. Also Consider Descriptive Analysis of Different Stackexchange Communities.


### Readability Indices

Flesch-Kincaid Grade

Gunning Fog Index

Coleman-Liau Index (Using)

SMOG Index

Automated Readability Index

Flesch-Kincaid Reading Ease (Using)

Spache Score

New Dale-Chall Score (Using)

## Other Measures
Code Count (whether code is present)

Latex Count (whether Latex code is present)

Punctuation Count (how much punctuation is present)

Cleaned Text (Usable for sentiment analysis)


## Resources
<!-- We can throw links to data and other things here -->


### Datasets

[Kaggle](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags/data)

[Internet Archive](https://archive.org/details/stackexchange)
Currently testing work on AI and IOT data sets from Internet Archive. Should add more to extend analysis once we have a working model.

<!-- Evaulation for Grade
- type of dataset is important. better not to use kaggle
- do more feature engineering/data munging if you're using a cleaned up data set 
- get good results (better accuracy - it matters)
--!>
