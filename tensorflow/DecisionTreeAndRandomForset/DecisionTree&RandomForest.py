#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:08:34 2019

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['target'])
iris_data = pd.concat((x, y), axis=1)
iris_data = iris_data[['sepal length (cm)', 'petal length (cm)', 'target']]
iris_data = iris_data[iris_data['target'].isin([0, 1])]

x_train, x_test, y_train, y_test = train_test_split(iris_data[['sepal length (cm)', 'petal length (cm)']], iris_data['target'], test_size=0.3, random_state=0)

tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
history = tree.fit(x_train, y_train)
history.predict(x_test)

history.score(x_test, y_test)

export_graphviz(tree, 'tree.dot', feature_names=['sepal length (cm)', 'petal length (cm)'])

# Random Forest

forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, n_jobs=2)
forest.fit(x_train, y_train)
forest.predict(x_test)
forest.score(x_test, y_test)