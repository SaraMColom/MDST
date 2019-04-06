# Bag of words

# Chi test and linear ridge regression

#########################
# Import packages
#########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import nltk
import csv
import os
import string
]#nltk.download()
import nltk.classify.util
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
stop_words=set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import vaderSentiment as VS
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pillow
import seaborn
import wordcloud
import re

#########################
# Read in and Prepare data 
#########################

# Change directory
os.chdir('/Users/sara/Desktop/drugsCom_raw') 

# Open and explore testing and training data frame
datTrain=pd.read_table('drugsComTrain_raw.tsv')
datTrain.head() # First couple of rows
datTrain.shape # Data dimensions
datTrain.dtypes # class structure of data
datTrain.iloc[0,3] # Examine the first survey review

# Some summary stats
datTrain['rating'].describe()


# Construct a bag of words matrix.
# This will lowercase everything, and ignore all punctuation by default.
# It will also remove stop words.
vectorizer = CountVectorizer(lowercase=True, stop_words="english")

# We created our bag of words matrix with far fewer commands.
print(matrix.todense())

# Let's apply the same method to all the headlines in all reviews
# We'll also add the url of the submission to the end of the headline so we can take it into account.

#submissions['full_test'] = submissions["headline"] + " " + submissions["url"]
matrix = vectorizer.fit_transform(datTrain['review'])

print(matrix.shape)

# Chi square test to examine 'most informative words' 
###################################################


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Convert reviews variable to binary so it works with a chi-squared test.
col = datTrain['rating'].copy(deep=True)
col_mean = col.mean()
col[col < col_mean] = 0
col[(col > 0) & (col > col_mean)] = 1

# Find the 1000 most informative columns
selector = SelectKBest(chi2, k=1000)
selector.fit(matrix, col)
top_words = selector.get_support().nonzero()

# Pick only the most informative columns in the data.
chi_matrix = matrix[:,top_words[0]]

#########################
# Adding in more features
#########################

# Incorporate punctuation information

# Our list of functions to apply.
transform_functions = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("d", x)),
    lambda x: len(re.findall("[A-Z]", x)),
]

# Apply each function and put the results into a list.
columns = []
for func in transform_functions:
    columns.append(datTrain['review'].apply(func))
    # Convert the meta features to a numpy array.
meta = np.asarray(columns).T

# Convert the  dates column to datetime.

columns = []

review_dates = pd.to_datetime(datTrain["date"])

# Transform functions for the datetime column.
transform_functions = [
    lambda x: x.year,
    lambda x: x.month,
    lambda x: x.day,
]

# Apply all functions to the datetime column.
for func in transform_functions:
    columns.append(review_dates.apply(func))

# Convert the meta features to a numpy array.
non_nlp = np.asarray(columns).T

# Concatenate the features together.
features = np.hstack([non_nlp, meta, chi_matrix.todense()])

####################
# Making Predictions
####################
from sklearn.linear_model import Ridge
import random

train_rows = 129037 # ~ 80% of data set
# Set a seed to get the same "random" shuffle every time.
random.seed(1)

# Shuffle the indices for the matrix.
indices = list(range(features.shape[0]))
random.shuffle(indices)

# Create train and test sets.
train = features[indices[:train_rows], :]
test = features[indices[train_rows:], :]

train_review= datTrain['rating'].iloc[indices[:train_rows]]
test_review = datTrain['rating'].iloc[indices[train_rows:]]
train = np.nan_to_num(train)

# Run the regression and generate predictions for the test set.
reg = Ridge(alpha=.1)
reg.fit(train, train_review)

# Output details
#Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None,
 #  normalize=False, random_state=None, solver='auto', tol=0.001)

predictions = reg.predict(test)

# We're going to use mean absolute error as an error metric.
# Our error is about 2.076 review score, which means that, on average,
# our prediction is 2.076 review scores away from the actual number of review score.
print(sum(abs(predictions - test_review)) / len(predictions))

# As a baseline, we'll use the average number of review scores across all submissions.
# The error here is 2.82 -- our estimate is better, but not hugely so.
# There either isn't a ton of predictive value encoded in the
# data we have, or we aren't extracting it well.
average_review = sum(test_review)/len(test_review)
print(sum(abs(average_review - test_review)) / len(predictions))


# Chi test and linear ridge regression
