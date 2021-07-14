# -*- coding: utf-8 -*-
"""
Perceptron
"""

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = datasets.load_iris()
iris_data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
iris_data['target'] = iris['target']
iris_data = iris_data[['sepal length (cm)', 'petal length (cm)', 'target']]

iris_data = iris_data[iris_data['target'] != 2]
iris_data['target_class'] = iris_data['target'].map({
    0:1,
    1:-1
})
iris_data = iris_data[['sepal length (cm)', 'petal length (cm)', 'target_class']]

def sign(z):
    if z > 0:
        return 1
    else:
        return -1
    
w = np.array([0.,0.,0.])
error = 1
iterator = 0
while error != 0:
    error = 0
    for i in range(len(iris_data)):
        x,y = np.concatenate((np.array([1.]), np.array(iris_data.iloc[i])[:2])), np.array(iris_data.iloc[i])[2]
        if sign(np.dot(w,x)) != y:
            iterator += 1
            error += 1
            w = w + y*x            

print("iterator: "+str(iterator))
sns.lmplot('sepal length (cm)','petal length (cm)',data=iris_data, fit_reg=False, hue ='target_class')
# Decision boundary 的方向向量
x_decision_boundary = np.linspace(-0.5,7)
y_decision_boundary = (-w[1]/w[2])*x_decision_boundary - (w[0]/w[2])
plt.plot(x_decision_boundary, y_decision_boundary,'r')
plt.xlim(-0.5,7.5)
plt.ylim(5,-3)
plt.show()