# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:05:13 2016

@author: chyam
"""

#import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import analyticutils as au
import argparse

def main():
    
    argparser = argparse.ArgumentParser(description='This script will perform simple data analysis on a text classification dataset.')
    argparser.add_argument('-i', '--inFile', help='Input filename', required=True)
    argparser.add_argument('-o', '--outFile', help='Output filename', required=True)
    args = argparser.parse_args()
    
    # Load data
    #filename = "./data/clean-data-21092016.tsv"
    filename = args.inFile
    df = pd.read_csv(filename, delimiter='\t')
    
    # Unique samples per label; label_count is a series
    label_count = df['Label'].value_counts(); print(label_count)
    
    # Visualisation (Seaborn): sample count per label
    sb.set(style="whitegrid", color_codes=True)
    sb.countplot(x="Label", data=df)

    # Visualisation (plotly): pie
    au.draw_pie(label_count, args.outFile)
    
    # Just a message
    print("Data analysed .........")
    

if __name__ == '__main__':
    main()    

    
    
#==============================================================================
#     # Visualisation(matplotlib): sample count per label
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('Labels')
#     ax.set_ylabel('Counts for each label')
#     ax.set_title("Sample count per label")
#     label_count.plot(kind='bar')
#==============================================================================
