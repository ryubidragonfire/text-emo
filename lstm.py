# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:07:10 2016

@author: chyam
purpose: A vanila lstm model for text classification.
"""

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM

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
    
    ### LSTM
    nb_classes = len(le.classes_)
    lstm(X_train_array, y_train, X_test_array, y_test, timesteps=1, batch_size=50, nb_epoch=2, nb_classes=nb_classes)
 
    ### Clean up
    del X_train_array, X_test_array, y_train, y_test

    return
    
def lstm(X_train, y_train, X_test, y_test, timesteps, batch_size, nb_epoch, nb_classes):
    """ Building a lstm model."""
    print('Starting LSTM ...')
    print(str(datetime.now()))
    feature_len = X_train.shape; print(feature_len[1])
    model = Sequential()
    #model.add(Embedding(max_features, 256, input_length=maxlen))
    model.add(LSTM(input_dim=feature_len[1], output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    print('LSTM finished ...')
    print(str(datetime.now()))    
    return    
    
if __name__ == '__main__':
    main()