
# Work embedding & neuronal network analysis

#Import libraries
import os
import pandas as pd
from keras.models import Sequential
from keras import layers

# MLP 
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Steps

# Read in data
# Process reviews
# Binarize ratings
# Separate data into testing and training


# Read in data
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

# Lets do a random subset of data and then look at distribution
import random
random.seed(4)
RandomNums=np.random.choice(100000, 10000) # Sample of ten thousand samples

datSubset=datTrain.iloc[RandomNums]

# Assign reviews as negative or positive

datSubset['BinaryScore']=''
datSubset.loc[datSubset['rating']>=5,'BinaryScore']=0 # negative reviews
datSubset.loc[datSubset['rating']<5,'BinaryScore']=1 # positive reviews

# Clean up reviews (i.e. remove punctuation marks, lowercase)




# we want to use the binary cross entropy and 
# the Adam optimizer you saw in the primer mentioned before. 
# Keras also includes a handy .summary() function to give an overview 
# of the model and the number of parameters available for training:

# Split data into training and testing
reviews = datSubset['review'].values
y = datSubset['BinaryScore'].values
reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, y, test_size=0.25, random_state=1000)

# Encoding


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)



# Word embedding
# You can start by using the Tokenizer utility class 
# which can vectorize a text corpus into a list of integers. 
# Each integer maps to a value in a dictionary that encodes 
# the entire corpus, with the keys in the dictionary being 
# the vocabulary terms themselves. You can add the parameter 
# num_words, which is responsible for setting the size of the vocabulary. 
# The most common num_words words will be then kept.

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews_train)

X_train = tokenizer.texts_to_sequences(reviews_train)
X_test = tokenizer.texts_to_sequences(reviews_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(reviews_train[2])
print(X_train[2])

# One problem that we have is that each text sequence has 
# in most cases different length of words. To counter this, 
# you can use pad_sequence() which simply pads the sequence of
#  words with zeros. By default, it prepends zeros but we want to 
# append them. Typically it does not matter whether you prepend or 
# append zeros.

# Additionally you would want to add a maxlen parameter
#  to specify how long the sequences should be.

from keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

from keras.models import Sequential
from keras import layers

# Global max/average pooling takes the maximum/average of all features 
# whereas in the other case you have to define the pool size. Keras 
# has again its own layer that you can add in the sequential model:
embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

#Traceback (most recent call last):
  #File "<stdin>", line 1, in <module>
  #File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/keras/engine/network.py", line 1252, in summary
   # 'This model has not yet been built. '
#ValueError: This model has not yet been built. Build the model first by calling build() or calling fit() with some data. Or specify input_shape or batch_input_shape in the first layer for automatic build.


history = model.fit(X_train, y_train,epochs=100,verbose=False,validation_data=(X_test, y_test),batch_size=10)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)