# Load libraries
import sys
import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

#Function that defines useful patterns for solving the problem
def patterns(name):
    return {
        'first': name[0], # First letter of the name
        'first2': name[0:2], # First 2 letters of the name
        'first3': name[0:3], # First 3 letters of the name
        'last': name[-1], #Last letter of the name
        'last2': name[-2:], #Last 2 letters of the name
        'last3': name[-3:], #Last 3 letters of the name
    }

#This vectorizes the patterns function in order to make easier to handle 
patterns = numpy.vectorize(patterns)

#This function sets the whole model
def setModel(path, test_size):
    # Loading dataset
    names = ['Index', 'Name', 'Gender']
    dataset = pandas.read_csv(path,names=names)
    dataset['Name'] = dataset['Name'].str.lower()

    #Obtaining Xs and Ys
    X = patterns(dataset['Name'])
    Y = dataset['Gender']

    # Split-out test and train datasets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=float(test_size))

    #Vectorizing the X values created by the patterns function
    vectorizer = DictVectorizer()
    vectorizer.fit(X_train)

    # Defining Decision Tree Classifier Algorithm on training dataset
    alg = DecisionTreeClassifier()
    alg.fit(vectorizer.transform(X_train), Y_train)

    # Returning all the relevant varibles for the model
    return alg, vectorizer, patterns, X_train, X_test, Y_train, Y_test

#Some strings for correct syntax helping
invalid_command = "   Please Please specify a valid command. Syntax: \n"
acc = "   python NameGender.py accuracy [path_to_dataset.csv] [test_size] \n"
pred = "   python NameGender.py predict [path_to_dataset.csv] [test_size] [names_to_predict.txt] \n"

#Main script
if(len(sys.argv) < 2):
    print invalid_command + acc + pred
elif(sys.argv[1] == "accuracy" and len(sys.argv) == 4):
    alg, vectorizer, patterns, X_train, X_test, Y_train, Y_test = setModel(sys.argv[2],sys.argv[3])
    print "Accuracy on Training Set: %f" % (alg.score(vectorizer.transform(X_train), Y_train))
    print "Accuracy on Testing Set: %f" % (alg.score(vectorizer.transform(X_test), Y_test))
elif(sys.argv[1] == "predict" and len(sys.argv) == 5):
    alg, vectorizer, patterns, X_train, X_test, Y_train, Y_test = setModel(sys.argv[2],sys.argv[3])
    names = open(sys.argv[4]).read().split()
    names_lowercase = [x.lower() for x in names]
    genders = alg.predict(vectorizer.transform(patterns(names_lowercase)))
    print "{:>12} {:>12}".format('Name', 'Gender')
    for i in xrange(len(names)):
        print "{:>12} {:>12}".format(names[i], genders[i])
else:
    print invalid_command + acc + pred

