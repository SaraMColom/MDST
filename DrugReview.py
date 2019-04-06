# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import csv
import os
import string
import nltk.classify.util
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
#nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import vaderSentiment as VS
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pillow
import seaborn as sns
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import wordcloud
import re

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

# Histogram
    # By rating
datTrain['rating'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.show()
    # By meds

sns.countplot(datTrain['drugName'], color='gray')

plt.show()       

# Preliminary test

# Assaign the rating to negative or positive

datTrain['BinaryScore']=''
datTrain.loc[datTrain['rating']>=5,'BinaryScore']=0 # negative reviews
datTrain.loc[datTrain['rating']<5,'BinaryScore']=1 # positive reviews

# Histogram
datTrain['BinaryScore'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')

plt.show() # Not balanced data between negative and positive reviews

# Notes:
# Remove Blank rows in Data, if any
# Remove punctuation
# Make lowercase
# Tokenize
# Remove stop words
# Word cloud on word frequency between negative and positive scores
# Study sparsity
# Examine distribution of quality scores


# Future

# Retain stop words for sentiment analysis
# Utilize word embedding vectors created already by wiki and google

#       IMPORTANT
# For sentiment analysis punctuation marks should not be removed will test this out later.

######################
# Version 1: Clean up text!!!!!!!
######################

# Drop blank rows
datTrain['review'].dropna(inplace=True)

# Lower case 
datTrain['review'] = [entry.lower() for entry in datTrain['review']]

# Remove punctuation

# Create fore loop to run thru string punctuations and remove it

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

remove_punctuations(datTrain.iloc[0,3]) # Example of result

# Overwrite the review column with text without punctuation
datTrain['review']=[entry.remove_punctuations() for entry in datTrain['review']]
datTrain['review_rmv']=remove_punctuations(datTrain['review']) # Created version with NO punctuation

# Tokenize 
#nltk.download('punkt') # Prior testing asked me to download this*
    
datTrain['review_rmv']=[word_tokenize(entry) for entry in datTrain['review_rmv']]

# Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# Commented out the removal of stopword/lementizer took too long to run

#for index,entry in enumerate(datTrain['review_rmv']):
    # Declaring Empty List to store the words that follow the rules for this step
#    Final_words = []
    # Initializing WordNetLemmatizer()
 #   word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
 #   for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
 #       if word not in stopwords.words('english') and word.isalpha():
 #           word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
 #           Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
 #   datTrain.loc[index,'text_final'] = str(Final_words)

# Make a filter function to remove stop words
def FilterStopWords(Input):
  filtered_sentence = [w for w in Input if not w in stop_words]
  for w in Input:
    if w not in stop_words:
        filtered_sentence.append(w)
        return(filtered_sentence)

# Test the function
TestFiltered=FilterStopWords(datTrain['review_rmv'].iloc[0])   

# Apply the function across the 'review' column
datTrain['review_rmvF']=[FilterStopWords(entry) for entry in datTrain['review_rmv']]

# Stem the words

st = PorterStemmer()

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [st.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


Test = datTrain['review_rmvF'].apply(str)
Test2 = Test.apply(stem_sentences)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Test2,datTrain['BinaryScore'],test_size=0.3)

# Encoding X and Y variables
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Test2)
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Tfidf_vect.vocabulary_) # corups
print(Train_X_Tfidf) # vectorized data

# Use the ML Algorithms to Predict the outcome

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

mat = confusion_matrix(Test_Y, predictions_NB)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,cmap='plt.cm.Blues
            #xticklabels=train.target_names, yticklabels=train.target_name
            )
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

# Remove words one character longs
#shortword = re.compile(r'\W*\b\w{1,3}\b') # Function for removing
#stringWords=shortword.sub('', stringWords)

# Calculate frequency of ea word
all_words = nltk.FreqDist(Test2)

ç= list(all_words.keys())[:3000]


# Subset data into negative and positive reviews
NegativeWords=datTrain[datTrain['BinaryScore']==0]
PositiveWords=datTrain[datTrain['BinaryScore']!=0]

# Collapse reviews
#stringNeg = ''.join(NegativeWords['FilterReview'])

# Convert to list
Negative=tuple(NegativeWords.review_rmvF.tolist())
Positive=tuple(PositiveWords.review_rmvF.tolist())

# Convert to string
stringNeg = str(Negative)
stringPos = str(Positive)

# Examine sparsity of negative and positive words

# Remove sparse words

# Set parameters for word cloud

# Create and generate a word cloud image:
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Negaive words
wordcloud = WordCloud(background_color="darkblue").generate(stringNeg)
plt.imshow(wordcloud, interpolation='bilinear')

# Display the generated image:
plt.axis("off")
plt.show()

# Positive words
wordcloud = WordCloud(background_color="darkblue").generate(stringPos)
plt.imshow(wordcloud, interpolation='bilinear')

# Display the generated image:
plt.axis("off")
plt.show()

# Feature extraction
# creating the feature matrix 
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=1000)
X = matrix.fit_transform(StringReview).toarray()

# Add classes to variables


# Start with one review:

# Examine distribution of quality scores
# Binaraize scores / color word cloud by Negative or positive reviews
