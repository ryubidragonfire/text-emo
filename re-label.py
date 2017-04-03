# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 08:44:56 2016

@author: chyam
Purpose: Re-label classes for text classification.
"""
import argparse
import pandas as pd
import preputils as pu

def main():
    argparser = argparse.ArgumentParser(description='This script will re-label classes of a text classification dataset.')
    argparser.add_argument('-i', '--inFile', help='Input filename', required=True)
    argparser.add_argument('-o', '--outFile', help='Output filename', required=True)
    args = argparser.parse_args()

    ### Load data
    df = pd.read_csv(args.inFile, delimiter='\t'); print(df.shape)
    
    ### change labels: 
#==============================================================================
#['attractiveness', 0     -> 0
# 'curiosity',      1     -> discard
# 'disgust',        2     -> 2
# 'fear',           3     -> 2
# 'germanangst',    4     -> 2
# 'happiness',      5     -> 0 
# 'indulgence',     6     -> 0
# 'neutral',        7     -> 1
# 'sadness',        8     -> 2
# 'surprise'        9]    -> discard
#==============================================================================

    sub_df = df[(df.Label != 'curiosity') & (df.Label != 'surprise')]; print(sub_df.shape)
    sub_df.loc[(sub_df.Label == 'attractiveness') | (sub_df.Label == 'happiness') | (sub_df.Label == 'indulgence'), 'Label'] = 'happy'
    sub_df.loc[(sub_df.Label == 'disgust') | (sub_df.Label == 'fear') | (sub_df.Label == 'germanangst') | (sub_df.Label == 'sadness'), 'Label'] = 'unhappy'

    pu.save_data(sub_df, args.outFile)
    
    # Just a message
    print("Re-labelled and saved............")
    
if __name__ == '__main__':
    main()
