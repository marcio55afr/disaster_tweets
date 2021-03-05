# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:09:17 2021

@author: marcio & tiago
"""

#### libs about

#structure and data
import numpy as np
import pandas as pd
import seaborn as sns
import math

#plot and images
from matplotlib import pyplot as plt

#text and mining
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def main():
    
    # Data to train and evaluate a model on Kaggle
    train, raw_test = readTweets()
    y = train.target
    x = train.drop('target')
    
    # Split the train with 20% and shuffle because of
    # the data is sorted by the column "keyword"
    train, test = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)
    
    
    
    vectorizer = CountVectorizer()
    




    scores = cross_val_score(model, X, y, cv=5)
    print(scores)







## read data (may be different files in future)
def readTweets():
    train = pd.read_csv('data/input/train.csv')
    test = pd.read_csv('data/input/test.csv')
    return train, test
    










main()
















