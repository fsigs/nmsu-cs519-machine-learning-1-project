import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
# Function libraries to scalate, split, get accuracy, and load datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Our models to be utilized
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class AMLProject:

  def set_dataset(self):
    
    self.dataset = pd.read_csv("../datasets/telecom_churn_data.csv", header=None)
    # Clean object columns
    # Transform Null values
    
    self.X = self.dataset.iloc[:, :-1]
    self.y = self.dataset.iloc[:, -1:].values.ravel()
    
    sc = StandardScaler()
    sc.fit(self.X)
    self.X = sc.transform(self.X)
  
  def train_test_split(self, test_size=0.2, random_state=40):
   return train_test_split(self.X, self.y, test_size, random_state) 