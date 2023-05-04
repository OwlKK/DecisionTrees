from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data = load_iris()
x = data.data
y = data.target

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25)

# train classifier
dtc = DecisionTreeClassifier()
dtc.fit(xTrain, yTrain)

# make predictions
yTrainPredict = dtc.predict(xTrain)
yTestPredict = dtc.predict(xTest)

# accuracy of predictions
trainAccuracy = accuracy_score(yTrain, yTrainPredict)
testAccuracy = accuracy_score(yTest, yTestPredict)

print(f'Train accuracy: {trainAccuracy:.2f}')
print(f'Test accuracy: {testAccuracy:.2f}')

# 8  The result for the train data represents the accuracy of the model on the training data.
#   - How well the model is able to correctly predict the labels of the training data.

# random_state, values, mean, stddev

testAccuracies = []
for i in range(10):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25, random_state=i)
    dtc = DecisionTreeClassifier()
    dtc.fit(xTrain, yTrain)
    yTestPredict = dtc.predict(xTest)
    testAccuracy = accuracy_score(yTest, yTestPredict)
    testAccuracies.append(testAccuracy)

meanTestAccuracy = np.mean(testAccuracies)
stdDevTestAccuracy = np.std(testAccuracies)

print(f'Mean test accuracy: {meanTestAccuracy:.2f}')
print(f'Standard deviation of test accuracy: {stdDevTestAccuracy:.2f}')

# Test split ratios
splitRatios = [0.1, 0.2, 0.3, 0.4, 0.5]
testAccuracies = []
for splitRatio in splitRatios:
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=splitRatio)
    dtc = DecisionTreeClassifier()
    dtc.fit(xTrain, yTrain)
    yTestPredict = dtc.predict(xTest)
    testAccuracy = accuracy_score(yTest, yTestPredict)
    
    testAccuracies.append(testAccuracy)
    print(f'Test accuracy with split ratio {splitRatio}: {testAccuracy:.2f}')

# 11 Chart
plt.plot(splitRatios, testAccuracies)
plt.xlabel('Split ratio')
plt.ylabel('Test accuracy')
plt.show()

# Tree

clf = DecisionTreeClassifier()
clf.fit(xTrain, yTrain)

plt.figure(figsize=(12, 12))
plot_tree(clf, filled=True)
plt.show()
