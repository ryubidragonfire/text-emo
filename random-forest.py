# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:30:47 2016

@author: chyam
purpose: Random Forest on text classification. 
"""

import argparse
import pandas as pd
import preputils as pu
import analyticutils as au

#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import math

def main():
    argparser = argparse.ArgumentParser(description='This script will perform random forest on text classification.')
    argparser.add_argument('-i', '--inFile', help='Input filename', required=True)
    argparser.add_argument('-o', '--outFile', help='Output filename', required=True)
    argparser.add_argument('-n', '--ngramFrom', help='Range of ngram, from n to m', required=True)
    argparser.add_argument('-m', '--ngramTo', help='Range of ngram, from n to m', required=True)
    argparser.add_argument('-a', '--analyzer', help='Text vectorization: word or char', required=True)
    #argparser.add_argument('-n_estimators', '--n_estimators', help='Number of trees', required=True)
    argparser.add_argument('-p', '--plot', help='Visualisation: True or False', default=False)
    args = argparser.parse_args()
    n = int(args.ngramFrom)
    m = int(args.ngramTo)
    #n_estimators = int(args.n_estimators)
    plot = bool(args.plot)
    
    ### Load data
    df = pd.read_csv(args.inFile, delimiter='\t'); print(df.shape)

    ### Perpare label
    y, le = pu.prep_label(df); print(le.classes_); 

    ### Prepare features
    tfidf_vectorizer = TfidfVectorizer(analyzer=args.analyzer, ngram_range=(n,m), stop_words=None, max_df=0.9, min_df=1)
    X_tfidf = tfidf_vectorizer.fit_transform(df['Text'].values.astype('U')); print(X_tfidf.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, train_size=0.7, random_state=88)

    ### Visualisation: PCA
    plotname = args.outFile + '-' + args.analyzer + '-' + str(n) + str(m) + 'gram'
    if plot==True:
        au.draw_PCA(X_tfidf, y, plotname)
    
    def doRandomForest():
        print('n_estimators: ' + str(n_estimators))
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features=int(math.sqrt(X_tfidf.shape[1])), class_weight='balanced', max_depth=None,min_samples_split=2, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        ### Performance reporting
        target_names = le.classes_
    #    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    #    
    #    print('Confusion Matrix')
    #    print(metrics.confusion_matrix(y_test, y_pred))
    #    
    #    print('\n', 'Accuracy')
    #    print(metrics.accuracy_score(y_test, y_pred))
        
        filename = args.outFile + '-' + args.analyzer + '-' + str(n_estimators) + 'estimators-' + str(n) + str(m) + 'gram'
        au.classification_metrics2(y_test, y_pred, target_names=target_names, filename=filename)
        
        return
        
    for n_estimators in range(10,60,10): # 10, 20, 30, 40, 50
        doRandomForest()
  
if __name__ == '__main__':
    main()    
    
    ### Usage
    ### python random-forest.py -i "./data/clean-data-21092016-processed-on-11102016.tsv" -o "./results/random-forest" -n_estimators 10 -n 1 -m 1 -a char
    ### python random-forest.py -i "./data/clean-data-21092016-processed-on-11102016.tsv" -o "./results/random-forest" -n 1 -m 1 -a char
    ### python random-forest.py -i "./data/emo3-clean-data-21092016.tsv" -o "./results/emo3/random-forest" -n 1 -m 1 -a char

