import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PCA import PCA

class GaussianNaiveBayes():

    def __init__(self, classes) :
        self.classes = classes

    # this function fits the model based on training data
    def fit(self, X_train, y_train):
        self.instances = len(X_train)
        self.features = len(X_train[0])
        self.X_train = X_train
        self.y_train = y_train

        self.class_counts = np.zeros((self.classes))
        for i in range(self.instances): 
            self.class_counts[self.y_train[i]] += 1

    # function to calculate the probability of each class P(Y = yp)
    def getClassPr(self, yp):
        return self.class_counts[yp]/self.instances

    # function to cal P(xi|yp)
    def getP_x_given_y(self, ft, val, yp):
        count = 0
        for i in range(self.instances):
            if abs(self.X_train[i, ft] - val) < 0.1 and self.y_train[i] == yp:
                count += 1;
        
        return count/self.class_counts[yp]

    # function to predict given test data
    def predict(self, X_test):
        y_pred = []

        for i in range(len(X_test)):
            # calculate which class has max value of-  product P(xi/class_i) * p(class_i)
            pred_prob = 0
            pred_class = 0

            for j in range(self.classes):
                # loop over all features
                class_score = 1
                for k in range(self.features):
                    class_score *= self.getP_x_given_y(k, X_test[i, k], j)

                class_score *= self.getClassPr(j)

                if(class_score > pred_prob):
                    pred_prob = class_score
                    pred_class = j
                
            y_pred.append(pred_class)

        print(y_pred)
        return y_pred

def main():

    # Loading iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    # performing pca 
    pca = PCA()
    pca.fit(X_train)
    pca.fix_components(2)
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)

    # predicting using naive bayes classifier
    naive_bayes = GaussianNaiveBayes(3) # iris dataset has 3 classes
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)

    # analysing the metrics of the algorithm
    print("accuracy = " + str(metrics.accuracy_score(y_test, y_pred)))
    print("precision = " + str(metrics.precision_score(y_test, y_pred, average='macro')))
    print("recall = " + str(metrics.recall_score(y_test, y_pred, average='macro')))
    print("f1_score = " + str(metrics.f1_score(y_test, y_pred, average='macro')))


if __name__=="__main__":
    main()

                
        
        

