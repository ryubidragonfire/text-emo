# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:50:23 2016
@author: chyam
Purpose: Utilities for data preparation.
"""
from __future__ import print_function

import pandas as pd
import re
from sklearn import preprocessing

def load_german_emo(filename):
    df_all = pd.read_csv(filename)
    df = df_all[['Sentence:System.String', 'Label:System.String']]
    df.columns = ['Text', 'Label']; print("Original df.shape: " + str(df.shape))
    return df
    
def remove_unwanted_chars(df):
    for index, row in df.iterrows():
        string = row['Text']
        newString = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|RT"," ",string).split())
        df.Text[index] = newString 
    return df
    
def remove_duplicates(df):
    df = df.drop_duplicates(); print("After df.drop_duplicates(), df.shape: " + str(df.shape))
    return df
    
def tolowercase(df):
    df['Text'] = df['Text'].str.lower()
    return df
    
def clean_data(df):
    df = df.dropna(); print("After df.dropna(), df.shape: " + str(df.shape))
    df = remove_unwanted_chars(df)
    df = tolowercase(df)
    df = remove_duplicates(df)
    return df
       
def save_data(df, filename):
    df.to_csv(filename, sep='\t', cols=['Label', 'Text'], index=False)
    return
    
def prep_label(df):
    """ Convert categorical labels into numerical labels. """
    le = preprocessing.LabelEncoder()
    le.fit(df.Label);           #print(list(le.classes_))
    y = le.transform(df.Label); #print(y[0:5]); print(list(le.inverse_transform(y[0:5])))
    return y, le
    