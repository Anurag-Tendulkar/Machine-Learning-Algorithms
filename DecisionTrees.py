import math
from matplotlib import pyplot as plt
from cProfile import label
from matplotlib import pyplot as plt
from sklearn import tree

# This method returns shannon entropy given the class counts for a splitd
def calShannonEntropy(count_0, count_1):
    i_class_0 = 0 
    i_class_1 = 0
    if count_1 != 0:
        i_class_1 = count_1/(count_0 + count_1) * math.log2((count_0 + count_1)/ count_1)
    if count_0 != 0:
        i_class_0 = count_0/(count_0 + count_1) * math.log2((count_0 + count_1)/count_0)
    return i_class_0 + i_class_1


# function to calculate the count of class labels in both the sets formed by a split
def get_class_counts(split_val, x_s, x_e, y_s, y_e, toys, feature):
    # split_val is the value of f1/f2 atribute which is used for the split
    # x_s is the start value and x_e is the end value of the window we are currently in
    # following are the count of points belonging to classes in both sets of the split
    set0_class0 = 0
    set0_class1 = 0
    set1_class0 = 0
    set1_class1 = 0


    for toy in toys:
        # x is the value of f1 atribute of the point
        # y is the value of f2 atribute of the point
        x = toy.f1
        y = toy.f2

        if x < x_s or x > x_e or y < y_s or y > y_e :
            continue

        if feature == "f1":

            if x < split_val and toy.label == 0:
                set0_class0 += 1
            elif x < split_val and toy.label == 1:
                set0_class1 += 1
            elif x > split_val and toy.label == 0:
                set1_class0 += 1
            else:
                set1_class1 += 1

        else :

            if y < split_val and toy.label == 0:
                set0_class0 += 1
            elif y < split_val and toy.label == 1:
                set0_class1 += 1
            elif y > split_val and toy.label == 0:
                set1_class0 += 1
            else:
                set1_class1 += 1

    return (set0_class0, set0_class1, set1_class0, set1_class1)


# function to calculate the information gain
def calIGain(set0_class_0, set0_class_1, set1_class_0, set1_class_1, entropy):
    tot_classes = set0_class_0 + set0_class_1 + set1_class_0 + set1_class_1
    set0_entropy = calShannonEntropy(set0_class_0, set0_class_1)
    set1_entropy = calShannonEntropy(set1_class_0, set1_class_1)

    return ( entropy - ((set0_class_0 + set0_class_1) / (tot_classes) * set0_entropy + (set1_class_0 + set1_class_1) / (tot_classes) * set1_entropy), set0_entropy, set1_entropy)


# function to calculate the split recursively
def getSplit(toys, x_s, x_e, y_s, y_e, entropy): 

# base case when the split is pure 
  if(entropy == 0):
        print("pure split")
        return 

  maxIgain = -math.inf

  for i in range(x_s, x_e): 
    # for each split over each value of attribute f1
    # calculate the values of class_0 and class_1 labels in both sides of the split
    (set0_class_0, set0_class_1, set1_class_0, set1_class_1) = get_class_counts(i+0.5, x_s, x_e, y_s, y_e, toys, "f1")
    (Igain, set0_entropy, set1_entropy) = calIGain(set0_class_0, set0_class_1, set1_class_0, set1_class_1, entropy)
    if Igain > maxIgain:
        split = ("f1", i+0.5)
        maxIgain = Igain
        h1_entropy = set0_entropy
        h2_entropy = set1_entropy

  for j in range(y_s, y_e): # for each split over each value of attribute f2
    # calculate the values of class_0 and class_1 labels in top and bottom half
    (set0_class_0, set0_class_1, set1_class_0, set1_class_1) = get_class_counts(j+0.5, y_s, y_e, x_s, x_e, toys, "f2")
    (Igain, set0_entropy, set1_entropy) = calIGain(set0_class_0, set0_class_1, set1_class_0, set1_class_1, entropy)
    if Igain > maxIgain:
        split = ("f2", j+0.5)
        maxIgain = Igain
        h1_entropy = set0_entropy
        h2_entropy = set1_entropy

  # call getSplit in the two halves recursively
  (atribute, value) = split

  print(split, h1_entropy, h2_entropy)

  if atribute == "f1":

    x_values = [value, value]
    y_values = [y_s, y_e]
    plt.plot(x_values, y_values, color = "k")
    getSplit(toys, x_s, math.floor(value), y_s, y_e, h1_entropy)
    getSplit(toys, math.ceil(value), x_e, y_s, y_e, h2_entropy)

  else:

    x_values = [x_s, x_e]
    y_values = [value, value]
    plt.plot(x_values, y_values, color = "k")
    getSplit(toys, x_s, x_e, y_s, math.floor(value), h1_entropy)
    getSplit(toys, x_s, x_e, math.ceil(value), y_e, h2_entropy)

  return


# function to calculate gini index 
def calGiniIndex(count_0, count_1):
    return 1 - math.pow(count_0/(count_0 + count_1), 2) - math.pow(count_1/(count_0 + count_1), 2)


# This class contains information about the dataset
class Toy:

    def __init__(self, f1, f2, label) -> None:
        self.f1 = f1
        self.f2 = f2
        self.label = label
        
toys = [Toy(1, 1, 1), Toy(1, 2, 1), Toy(1, 3, 1), Toy(2, 1, 1), Toy(2, 2, 1), Toy(2, 3, 1), Toy(4, 1, 1), Toy(4, 2, 1), Toy(4, 3, 0), Toy(4, 4, 0), Toy(5, 1, 1), Toy(5, 2, 1), Toy(5, 3, 0), Toy(5, 4, 0)]
X = [[toy.f1, toy.f2] for toy in toys]
Y = [toy.label for toy in toys]


# This is the driver code 
# This code snippet is for plotting the toys and drawing the decision boundary
for toy in toys :
    x = toy.f1
    y = toy.f2
    label = toy.label
    if label == 1:
        plt.scatter([x], [y], marker = "x", color = "blue", label = "Class-1")
    else:
        plt.scatter([x], [y], marker = "o", color = "orange", label = "Class-0")

plt.xlabel("f1")
plt.ylabel("f2")

plt.xticks(range(0, 6, 1))
plt.yticks(range(0, 5, 1))

# from stackoverflow to prevent duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# get split of data
# get initial entropy
class_0 = 0
class_1 = 0
for toy in toys:
    
    if toy.label == 0:
        class_0 += 1
    else:
        class_1 += 1

    
inintial_SE = calShannonEntropy(class_0, class_1)
# function call for geting the decision boundaries for the data
getSplit(toys, 0, 6, 0, 5, inintial_SE)
plt.show()

# the code below is the decision tree for the data usink skylearn
# from package skylearn
clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
tree.plot_tree(clf)
plt.show()