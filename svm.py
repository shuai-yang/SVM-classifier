# Scikit-learn’s make_blobs function allows us to generate the two clusters/blobs of data
from sklearn.datasets import make_blobs
# Scikit-learn’s train_test_split function allows us to split the generated dataset into a part for training and a part for testing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Define configuration options
# The random seed for our blobs ensures that we initialize the pseudorandom numbers generator with the same start initialization
blobs_random_seed = 42
# The centers represent the (X, y) positions of the centers of the blobs we’re generating
centers = [(0,0), (5,5)]
# The cluster standard deviation shows how scattered the centers are across the two-dimensional mathematical space. 
# The lower, the more condensed the clusters are
cluster_std = 1
# The fraction of the test split shows what percentage of our data is used for testing purposes
frac_test_split = 0.33
# The number of features for samples shows the number of classes we wish to generate data for
# For binary classifier, that is 2 classes
num_features_for_samples = 2
# The number of samples in total shows the number of samples that are generated in total
num_samples_total = 1000

'''(1) Generating a dataset'''
# Call make_blobs with the configuration options
# Variable inputs will store the features and variable targets will store class outcomes
inputs, targets = make_blobs(n_samples = num_samples_total, centers = centers, n_features = num_features_for_samples, cluster_std = cluster_std)
#print(inputs)
#print(targets)

# Split the inputs and targets into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=frac_test_split, random_state=blobs_random_seed)
#print(X_test)
#print(X_train)
#print(y_test)
#print(y_train)

# Save and load temporarily (optional)
np.save('./data.npy', (X_train, X_test, y_train, y_test))
X_train, X_test, y_train, y_test = np.load('./data.npy', allow_pickle=True)
#print(np.save)
#print(X_test)
#print(X_train)
#print(y_test)
#print(y_train)
# Now, if you run the code once, then comment np.save, you’ll always have your code run with the same dataset

# Generate scatter plot for training data 
plt.scatter(X_train[:,0], X_train[:,1])
plt.title('Linearly separable data (using 2 features)')
plt.xlabel('X1')
plt.ylabel('X2')
#plt.show()

'''(2) Building the SVM classifier'''
from sklearn import svm
# Initialize SVM classifier
clf = svm.SVC(kernel='linear')
#print(clf)
# Fit training data to the classifier
clf = clf.fit(X_train, y_train)
#print(clf)

'''(3) Using SVM to predict new data samples'''
# Predict the test set
predictions = clf.predict(X_test)
#print(predictions)
# Generate confusion matrix
from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
print(matrix)
plt.title('Confusion matrix for my classifier')
plt.show()
