# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:02:21 2016

@author: chyam
purpose: A vanila mlp model for text classification.
"""

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import SGD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

import pandas as pd
from datetime import datetime

import preputils as pu

def main():
    
    ### Load data
    filename = "C:/git/german-emo/data/clean-data-21092016.tsv"
    df = pd.read_csv(filename, delimiter='\t'); print(df.shape)

    ### Perpare label
    y, le = pu.prep_label(df); 

    ### Prepare features (word-based) -> Split data into training and test sets  
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words=None, ngram_range=(1,1), max_df=0.9, min_df=1)
    X_tfidf_word_11gram = tfidf_vectorizer.fit_transform(df['Text'].values.astype('U')); print(X_tfidf_word_11gram.shape); #11468x26778
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf_word_11gram, y, test_size=0.3, train_size=0.7, random_state=88); del X_tfidf_word_11gram
    X_train_array = X_train.toarray(); del X_train
    X_test_array = X_test.toarray(); del X_test
    
    ### MLP
    nb_classes = len(le.classes_)
    #mlp(X_train_array, y_train, X_test_array, y_test)
    mlprelu(X_train_array, y_train, X_test_array, y_test)
    
    ### Clean up
    del X_train_array, X_test_array, y_train, y_test

    return

def mlprelu(X_train, y_train, X_test, y_test):
    print('Starting MLP-relu ...')
    print(str(datetime.now()))
    model = Sequential()
    feature_len = X_train.shape; print(feature_len[1])
    model.add(Dense(64, input_dim=feature_len[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(X_train, y_train, nb_epoch=100, batch_size=50)
    score, acc = model.evaluate(X_test, y_test, batch_size=50)
    print('================== model.evaluate =================')
    print('Test score:  ', score)
    print('Test accuracy:  ', acc)
    
    print('MLP-relu finished ...')
    print(str(datetime.now()))
    return    
    
def mlp(X_train, y_train, X_test, y_test):
    """ Building a mlp model."""
    print('Starting MLP ...')
    print(str(datetime.now()))
    feature_len = X_train.shape; print(feature_len[1])
    model = Sequential()
    model.add(Dense(64, input_dim=feature_len[1], init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #If using binary representation
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    #print(model.summary())
    
    model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)
    print('================== model.evaluate =================')
    print('Test score:  ', score)
    print('Test accuracy:  ', acc)
    print('MLP finished ...')
    print(str(datetime.now()))
    return    

if __name__ == '__main__':
    main()