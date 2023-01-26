'''
    This following code is for performing principal component analysis from scratch for a given dataset
    I have performed PCA on the iris dataset (https://archive.ics.uci.edu/ml/datasets/iris)
    and have reduced the dimension of the dataset from 4 to 2
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from statistics import mean
from copy import deepcopy
import numpy as np

class PCA:

    ''' This functions mean centers the training data, calculates the eigen values of the variance covariance matrix
        and prints them ''' 
    def fit(self, X_train):
        # calculate the no. of features and instances.
        self.features = len(X_train[0])
        self.instances = len(X_train[:, 0:1])

        X_train_copy = deepcopy(X_train)

        # mean center the training data
        for i in range(self.features):
            mean_feature = mean(X_train[: , i:i+1].T[0])
            for j in range(self.instances):
                X_train_copy[j][i] = X_train_copy[j][i] - mean_feature
        
        # calculate the variance-covariance matrix
        VarCov = np.dot(X_train_copy.T, X_train_copy)/(self.instances - 1)

        # calculate the eigen vectors of the matrix
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(VarCov)

        # print(self.eigen_vals)

    # This function reduces the dimensionality of the data set to the specified value
    def fix_components(self, n_features):
        # calculate the feature matrix based on n_features
        self.feature_matrix = self.eigen_vecs[:, 0:n_features]
    
    '''This function transforms the original dataset into the principal components as 
       specified by fix_components function'''
    def transform(self, Data):
        # mean center the data 
        # calculate instances as Data may be training data or testing data that has to be transformed
        instances = len(Data[:, 0:1])
        Data_copy = deepcopy(Data)
        for i in range(self.features):
            feature_mean = mean(Data[:, i:i+1].T[0])
            for j in range(instances):
                Data_copy[j][i] = Data_copy[j][i] - feature_mean
        
        return np.dot(self.feature_matrix.T, Data_copy.T).T




# Defining main function
def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print(X_train)

    c0_train, c1_train, c2_train = 0, 0, 0

    for class_label in y_train:
        if class_label == 0:
            c0_train  += 1
        elif class_label == 1:
            c1_train += 1
        else: 
            c2_train += 1

    print("class 0 - train  " + str(c0_train) + " class 0 - test  " + str(50-c0_train))
    print("class 1 - train  " + str(c1_train) + " class 1 - test  " + str(50-c1_train))
    print("class 2 - train  " + str(c2_train) + " class 2 - test  " + str(50-c2_train))
  
    pca = PCA()

    pca.fit(X_train)

    pca.fix_components(2)

    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)

    # print(X_train_new)
    # print(X_test_new)




  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()