# Name: HCV Dataset Data Mining
# Author: Brianna Drew
# Date Created: October 8th, 2020
# Last Modified: October 12th, 2020

# importing all required libraries
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('https://raw.githubusercontent.com/briannadrew/hcvdat/main/hcvdat0.csv') # reading in the csv file

# SCATTERPLOT MATRIX
index_vals = df['Category'].astype('category').cat.codes # defining the class of the dataset and its different values (categories)
# creating a scatterplot matrix
fig = go.Figure(data=go.Splom(
    dimensions=[dict(label='Age',
                     values=df['Age']),
                dict(label='Sex',
                     values=df['Sex']),
                dict(label='ALB',
                     values=df['ALB']),
                dict(label='ALP',
                     values=df['ALP']),
                dict(label='ALT',
                     values=df['ALT']),
                dict(label='AST',
                     values=df['AST']),
                dict(label='BIL',
                     values=df['BIL']),
                dict(label='CHE',
                     values=df['CHE']),
                dict(label='CHOL',
                     values=df['CHOL']),
                dict(label='CREA',
                     values=df['CREA']),
                dict(label='GGT',
                     values=df['GGT']),
                dict(label='PROT',
                     values=df['PROT'])],
    text=df['Category'], # axis labels
    marker=dict(color=index_vals, # dots are colour-coded according to their class value
                showscale=True, # show scale defining the colour-coding based on class
                line_color='white', line_width=0.5) # defining colour and width of grid lines
    ))
fig.update_layout(
    title='HCV Dataset', # set title of graph
    dragmode='select', # so you can drag elements of the graph to get a customizable view
    width=3600, # width of graph
    height=3600, # height of graph
    hovermode='closest', # so you can view details of plotpoints when you hover over them
)
fig.show() # display the graph

# DATA PREPROCESSING
df.dropna(inplace = True) # drop all rows in data which contain missing values
l1 = LabelEncoder() # create instance of LabelEncoder
l1.fit(df['Sex'])
df.Sex = l1.transform(df.Sex) # transform values of "Sex" categorical attribute to numerical
norm = Normalizer() # create instance of Normalizer
df.iloc[:,:-1] = norm.fit_transform(df.iloc[:,:-1]) # normalize the data

# DECISION TREE & 10-FOLD CROSS-VALIDATION
X = (df.drop('Category', axis = 1)) # non-class data
y = df['Category'] # classes

myScores1 = cross_val_score(DecisionTreeClassifier(), X,y, scoring = 'accuracy', cv = KFold(n_splits = 10)) # create decision tree with 10-fold cross-validation and return accuracy scores
print(myScores1) # print resulting accuracy scores from cross-validation

# 3 KNN CLASSIFIER & 10-FOLD CROSS-VALIDATION
myScores2 = cross_val_score(KNeighborsClassifier(n_neighbors = 3), X,y, scoring = 'accuracy', cv = KFold(n_splits = 10)) # create k=3 KNN classifier with 10-fold cross-validation and return accuracy scores
print(myScores2) # print resulting accuracy scores from cross-validation
