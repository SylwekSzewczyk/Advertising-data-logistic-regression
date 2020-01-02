# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:20:15 2020

@author: Sylwek Szewczyk
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class LogRegression:
    
    def __init__(self, df):
        self.df = df
        self.X = self.df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
        self.y = self.df['Clicked on Ad']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regressor = False
        self.y_preds = None
        
    def showData(self):
        print(self.df.describe())
    
    def analyzeData(self):
        sns.set_palette("husl")
        return sns.pairplot(self.df, hue= "Clicked on Ad")
    
    def solve (self, testsize, random):
        if random < 0:
            raise ValueError('random_state must be greater than 0!')
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = testsize, random_state = random)
            self.regressor = LogisticRegression()
            self.regressor.fit(self.X_train, self.y_train)
            self.y_preds = self.regressor.predict(self.X_test)
        
    def report(self):
        if self.regressor:
            print(classification_report(self.y_test, self.y_preds))
        else:
            raise Exception("You need to solve the model first!")    
    
    @classmethod
    def getData(cls, data):
        return cls(df = pd.read_csv(data))
    

lr = LogRegression.getData('advertising.csv')
