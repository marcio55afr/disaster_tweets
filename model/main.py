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
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#classifiers
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression


def main():
    
    # Data to train and evaluate a model on Kaggle
    train, raw_test = readTweets()
    y_train = train.target
    X_train = train.drop('target',axis=1)
    
    # Split the train with 20% and shuffle because of
    # the data is sorted by the column "keyword"
    #train, test = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)
    
    
    # Transform the text content into one-hot encode
    vectorizer = CountVectorizer()
    text_vector = vectorizer.fit_transform(X_train.text)
    #print(vectorizer.get_feature_names())
    print(text_vector.toarray())


    ###### Classifiers, this will be in other functions later ######

    if(1):
        # 1-10 NN with cross-validation 5-folds
        print("\nNEAREST NEIGHBORS\n")
        for k in range(9):
            model = neighbors.KNeighborsClassifier(k + 1, weights='distance')
        
            scores = cross_val_score(model, text_vector, y_train, cv=5)
            print(scores, np.mean(scores))

    if(1):
        # LogisticRegression with cross-validation 5-folds
        print("\nLOGISTIC REGRESSION\n")
        params = {
            'l1':['liblinear','saga'],
            'l2':['newton-cg','sag','saga','lbfgs'],
            'none':['newton-cg','sag','saga','lbfgs']
            }
        
        for penalty_ in params.keys():
            for solver_ in params[penalty_]:
                model = LogisticRegression(penalty=penalty_, 
                                   random_state=42,
                                   solver=solver_,
                                   multi_class='ovr',
                                   n_jobs=10)
            
                scores = cross_val_score(model, text_vector, y_train, cv=5)
                print(scores, np.mean(scores))
    
    
    ########### Processing the text  ################
    
    # These preprocessings lead to a lower accuracy
    
    # lower case
    X_train.text = X_train.text.str.lower()
    
    # remove pontuation (# may has some weight on the classification)
    X_train.text = X_train.text.str.replace(r'[^\w\s]+', '')
                    
    # remove spaces
    X_train.text = X_train.text.str.strip()
    
    # remove number (Numbers may represent something or no)    
    X_train.text = X_train.text.str.replace(r'\d+', '')
    
    #### removing nonknowing words
    
    
    # Vector one-hot with english stop words
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    text_vector = vectorizer.fit_transform(X_train.text)
                    

    ###### Classifiers after that text processings ######

    if(1):
        # 1-10 NN with cross-validation 5-folds
        print("\nNEAREST NEIGHBORS\n")
        for k in range(9):
            model = neighbors.KNeighborsClassifier(k + 1, weights='distance')
        
            scores = cross_val_score(model, text_vector, y_train, cv=5)
            print(scores, np.mean(scores))


    if(1):
        # LogisticRegression with cross-validation 5-folds
        print("\nLOGISTIC REGRESSION\n")
        params = {
            'l1':['liblinear','saga'],
            'l2':['newton-cg','sag','saga','lbfgs'],
            'none':['newton-cg','sag','saga','lbfgs']
            }
        
        for penalty_ in params.keys():
            for solver_ in params[penalty_]:
                model = LogisticRegression(penalty=penalty_, 
                                   random_state=42,
                                   solver=solver_,
                                   multi_class='ovr',
                                   n_jobs=10)
        
                scores = cross_val_score(model, text_vector, y_train, cv=5)
                print(scores, np.mean(scores))


## read data (may be different files in future)
def readTweets():
    train = pd.read_csv('data/input/train.csv')
    test = pd.read_csv('data/input/test.csv')
    return train, test
    










main()
















