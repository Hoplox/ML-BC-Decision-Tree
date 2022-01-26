# Using a decision tree on the breast cancer dataset.
#
# Code is based on:
#
# http://scikit-learn.org/stable/modules/tree.html)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

#Load the data
bc = load_breast_cancer()
my_data = bc.data
my_target = bc.target

#80% in x,y train and 20% is in x,y test.
X_train, X_test, y_train, y_test = train_test_split(my_data, my_target, test_size=0.2, random_state=0)

dtree = tree.DecisionTreeClassifier(criterion="entropy")

#Train
dtree.fit(X_train,y_train)

#Print accuracy score of decision tree
print(dtree.score(X_test, y_test))