# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:01 2016
@author: chyam
Purpose: Prepare data for training and testing a german-emo model.
"""

import preputils as pu
import argparse

def main():
    argparser = argparse.ArgumentParser(description='This script will prepare data for text classifciation.')
    argparser.add_argument('-i', '--inFile', help='Input filename', required=True)
    argparser.add_argument('-o', '--outFile', help='Output filename', required=True)
    args = argparser.parse_args()
    
     # load raw data
    #df = pu.load_german_emo('./data/sentenceLabelsProd21092016.csv')
    df = pu.load_german_emo(args.inFile)
    
    # Clean data
    df = pu.clean_data(df)
    
    # Save data
    #pu.save_data(df, './data/clean-data-21092016-processed-on-10112016.tsv')
    pu.save_data(df, args.outFile)
    
    # Just a message
    print("Cleaned and saved............")
   
if __name__ == '__main__':
    main()
    