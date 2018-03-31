# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:13:04 2018

@author: mahmoudabdelfatahabd
"""
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
#for open csv file 
import re

#path of csv file
import os
path = r"E:\4th Year\GP\Data"

# Now change the directory
os.chdir( path )

#read csv file
dataset = pd.read_csv('data_set.csv', encoding = "ISO-8859-1")

#cleaning the text
stop_words = list(get_stop_words('en'))  
nltk_words = list(stopwords.words('english'))
stop_words.extend(nltk_words)
ps = PorterStemmer()
#get number of rows
#num_of_rows = int(dataset.count(0)['mood'])
num_of_rows = int(dataset.count(0)['title'])

filtered_lyric = []
for i in range(0, num_of_rows):
    #get each lyric and remove the new line space
    lyric = dataset['lyrics'][i]
    lyric = str(lyric)
    lyric = lyric.replace("\\n", ' ')
    lyric = re.sub('[^a-zA-Z]', ' ', lyric)
    lyric = lyric.lower()
    lyric = lyric.split()
    lyric = [ps.stem(w) for w in lyric if not w in stop_words]
    #convert list of word to paragraph
    lyric = ' '.join(lyric)
    filtered_lyric.append(lyric)
    #print(filtered_lyric)
    #filtered_lyric.clear()

tokenizer = lambda text: text.split()
tokenizer_porter = lambda text: [ps.stem(word) for word in text.split()]


### feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(binary=False,
                       stop_words=stop_words,
                       ngram_range=(1,1),
                       preprocessor=lambda text: re.sub('[^a-zA-Z]', ' ', text.lower()),
                       tokenizer=lambda text: [ps.stem(word) for word in text.split()])

#X_train_feat = cv.fit_transform(X_train, y_train).toarray().astype(np.int64)
X_train_feat = cv.fit_transform(filtered_lyric).toarray()
y = dataset.iloc[:, 4].values
                
#Splitting  the dataset into the Training Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_feat, y, test_size=0.20 )

################################################################################
#Support Vector Machine (SVM)
from sklearn.svm import SVC
clfSVM = SVC(kernel = 'linear')
clfSVM.fit(X_train, y_train)

#predicting the Test set results
y_pred = clfSVM.predict(X_test)

################################################################################
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
    
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)

print(acc*100)
    

        






