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

