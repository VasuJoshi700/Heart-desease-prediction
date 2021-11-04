# Heart-desease-prediction
# Import packages 
**import following packages and module that are useful for the operation    
import os     
import pandas as pd    
import numpy as np    
import matplotlib.pyplot as plt      
import scipy.stats as sc     
import seaborn as sns    
from sklearn import metrics   
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV      
from sklearn.preprocessing import StandardScaler, minmax_scale,MinMaxScaler        
from sklearn.neighbors import KNeighborsClassifier       
import warnings      
warnings.filterwarnings("ignore")     
from sklearn.ensemble import RandomForestClassifier    
from sklearn.tree import DecisionTreeClassifier     
from sklearn.linear_model import LogisticRegression**    

# Load the Dataset
Extracting the sumary of dataset
![heart summary](https://user-images.githubusercontent.com/87512268/135963835-5a613a08-0d88-423a-bade-f86f3c5a37f4.png)

# Investigate the outliers 
![heart boxplot](https://user-images.githubusercontent.com/87512268/135964641-ed06bfc8-61ff-4a1c-9152-e30362ed0aad.png)
In above boxplot we can see that there are outliers in cholestrol
The highest cholestrol level that one can have is 200-350 (taking worst case scenario), i capped the outlier at 350 mg/dl cholestrol

# Investigating null values
There are not any null value in variables 

# Exploratory Analysis

![heart age](https://user-images.githubusercontent.com/87512268/135967138-53403b8f-a906-4eb6-ab76-61d60124ee6f.png)
![heart dist age](https://user-images.githubusercontent.com/87512268/135967607-8de14415-7454-4d65-8c0c-86e86ef4087c.png)

In above figure we can observe the distribution of age in data set , people at the age between 58 to 60 are frequent in dataset

![heart cholestrol](https://user-images.githubusercontent.com/87512268/135967800-54fc10fd-3806-4dd5-8330-dbe6b605ea84.png)
From the above visuals we can perceive that people having cholestrol between 200 to 250 are frequent in dataset

![heart cholestrol and age](https://user-images.githubusercontent.com/87512268/135968260-477537e0-fbe4-483e-9ad0-5bca6b9e1cc6.png)
In the above regression plot we can perceive that there is weak positive correlation between age and cholestrol .

![heart thalach](https://user-images.githubusercontent.com/87512268/136006671-f2bade86-9266-4716-8a19-9a3e71ead486.png)
From the above graph we can observe the relationship between thalach and old_peak

![pie heart](https://user-images.githubusercontent.com/87512268/136007253-498a8993-996e-4010-83c9-9cb18095c9a3.png)   
 Above i check the balance of targeted class so this class has a balance ratio  .
 
# _Machine Learning Modelling_
**Divide the data into X and y (independent and targeted variable)**

### Standardise the data by using MinMaxScaler

MinMaxScaler. For each value in a feature, MinMaxScaler subtracts the minimum value in the feature and then divides by the range. The range is the difference between the original maximum and original minimum. MinMaxScaler preserves the shape of the original distribution.


# Performed Algorithims 
### KNN
![heart knn](https://user-images.githubusercontent.com/87512268/136008715-62dbd764-b42e-479c-a989-8f14bcef2971.png)
From the above graph i have choose the nearest neighbour which is 11,14,16
Check the score of the model 

I also performed hyperparameter tuning in models given below to stablize them and get the best results out of them 

### Logistic Regression
 In logistic regression, we decide a probability threshold. If the probability of a particular element is higher than the probability threshold then we classify that element in one group or vice versa.

### Decesion Tree
    Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).

In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

 
Gold BlogGold BlogDecision Tree Algorithm, Explained
<= Previous postNext post =>
 
 



Share542

Tags: Algorithms, Decision Trees, Explained

All you need to know about decision trees and how to build and optimize decision tree classifier.

SAS is a Leader in the
Magic Quadrant for
Data Quality Solutions
Read the Gartner Report



By Nagesh Singh Chauhan, Data Science Enthusiast.
comments


 

Introduction
 
Classification is a two-step process, learning step and prediction step, in machine learning. In the learning step, the model is developed based on given training data. In the prediction step, the model is used to predict the response for given data. Decision Tree is one of the easiest and popular classification algorithms to understand and interpret.

 

Decision Tree Algorithm
 
Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).

In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

 

Types of Decision Trees
 
Types of decision trees are based on the type of target variable we have. It can be of two types:

Categorical Variable Decision Tree: Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.
Continuous Variable Decision Tree: Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree.
Example:- Let’s say we have a problem to predict whether a customer will pay his renewal premium with an insurance company (yes/ no). Here we know that the income of customers is a significant variable but the insurance company does not have income details for all customers. Now, as we know this is an important variable, then we can build a decision tree to predict customer income based on occupation, product, and various other variables. In this case, we are predicting values for the continuous variables.

 

Important Terminology related to Decision Trees
 

Root Node: It represents the entire population or sample and this further gets divided into two or more homogeneous sets.
Splitting: It is a process of dividing a node into two or more sub-nodes.
Decision Node: When a sub-node splits into further sub-nodes, then it is called the decision node.
Leaf / Terminal Node: Nodes do not split is called Leaf or Terminal node.
Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say the opposite process of splitting.
Branch / Sub-Tree: A subsection of the entire tree is called branch or sub-tree.
Parent and Child Node: A node, which is divided into sub-nodes is called a parent node of sub-nodes whereas sub-nodes are the child of a parent node.


Decision trees classify the examples by sorting them down the tree from the root to some leaf/terminal node, with the leaf/terminal node providing the classification of the example.

Each node in the tree acts as a test case for some attribute, and each edge descending from the node corresponds to the possible answers to the test case. This process is recursive in nature and is repeated for every subtree rooted at the new node.

 

Assumptions while creating Decision Tree
 
Below are some of the assumptions we make while using Decision tree:

In the beginning, the whole training set is considered as the root.
Feature values are preferred to be categorical. If the values are continuous then they are discretized prior to building the model.
Records are distributed recursively on the basis of attribute values.
Order to placing attributes as root or internal node of the tree is done by using some statistical approach.
Decision Trees follow Sum of Product (SOP) representation. The Sum of product (SOP) is also known as Disjunctive Normal Form. For a class, every branch from the root of the tree to a leaf node having the same class is conjunction (product) of values, different branches ending in that class form a disjunction (sum).

The primary challenge in the decision tree implementation is to identify which attributes do we need to consider as the root node and each level. Handling this is to know as the attributes selection. We have different attributes selection measures to identify the attribute which can be considered as the root note at each level.


 
The decision of making strategic splits heavily affects a tree’s accuracy. The decision criteria are different for classification and regression trees.

Decision trees use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that the purity of the node increases with respect to the target variable. The decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.

The algorithm selection is also based on the type of target variables. 
 
### Random Forest 
### XG-Boost 
### ADA-Boost
Every model is working efficiently except XG-boost reason may be the low amount of data .





