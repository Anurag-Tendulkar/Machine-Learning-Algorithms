# **Machine Learning Algorithms**
 This repository contains machine learning algorithms that I have coded as a part of the ML course in BITS Goa

## **Results of Algorithms**
### **1. Decision Trees**
The following code builds a decision tree from scratch to classify the data below. The best split is decided by the gain in information after the split. A drop in entropy results in information gain. The algorithm tries to spit the data into pure halves at every split

<img width="400" alt="image" src="https://user-images.githubusercontent.com/74342035/214771062-b7adcfef-7afb-4fd3-b395-e34bc16b9675.png">

### **2. PCA**
The following code is for performing principal component analysis from scratch for any given dataset. I have performed PCA on the iris dataset (https://archive.ics.uci.edu/ml/datasets/iris) and have reduced the dimension of the dataset from 4 to 2.<br>
The eigen values of the variance covariance matrix of the iris dataset are - 4.15384465, 0.25811933, 0.08303984, 0.02201228. <br>
The maximum variance is captured by the first two principal components. Hence I have kept them and discarded the other two.
