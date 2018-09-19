# Load libraries
import pandas
import numpy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Defining the function that makes names processable
def name_transformation(name):
    arr = numpy.zeros(52)
    for ind, x in enumerate(name):
        arr[ord(x)-ord('a')] += 1
        return arr

# Loading dataset
path = "/home/dsalazar/Machine-Learning/NameGender/dataset/yob2010.txt"
names = ['Name', 'Gender', 'Count']
dataset = pandas.read_csv(path,names=names)
dataset['Name'] = dataset['Name'].str.lower()
dataset = dataset.drop(dataset[dataset.Count < 20].index)

#name_map returns an array of properties taking a name as input
name_map = numpy.vectorize(name_transformation, otypes=[numpy.ndarray])

#Obtaining Xs and Ys
Xname = dataset['Name']
Xlist = name_map(dataset['Name'])
X = numpy.array(Xlist.tolist())
Y = dataset['Gender']

# Split-out validation dataset
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('RFC', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions on validation dataset
svm = DecisionTreeClassifier()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

idx = numpy.random.choice(numpy.arange(len(Xlist)), 10, replace=False)
xs = Xname[idx]
ys = Y[idx]
pred = svm.predict(X[idx])
for a,b, p in zip(xs,ys, pred):
    print a,b, p
