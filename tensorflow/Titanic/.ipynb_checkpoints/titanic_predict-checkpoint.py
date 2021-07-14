#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:55:51 2019

@author: daniel

Kaggle - Titanic 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.getcwd()

train = pd.read_csv('titanic//train.csv')
test = pd.read_csv('titanic//test.csv')
submit = pd.read_csv('titanic//gender_submission.csv')

data = train.append(test)
data.reset_index(drop=True, inplace=True)

sns.countplot(data['Survived'])
sns.countplot(data['Pclass'], hue=data['Survived'])
sns.countplot(data['Sex'], hue=data['Survived'])
sns.countplot(data['Embarked'], hue=data['Survived'])
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Age', kde=False)
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Fare', kde=False)
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Parch', kde=False)
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'SibSp', kde=False)
data['Family_Size'] = data['Parch'] + data['SibSp']
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Family_Size', kde=False)

data['Title1'] = data['Name'].str.split(', ', expand=True)[1]
data['Title1'] = data['Title1'].str.split('.', expand=True)[0]
data_uq = data['Title1'].unique()
pd.crosstab(data['Title1'], data['Sex']).T.style.background_gradient(cmap='summer_r')

