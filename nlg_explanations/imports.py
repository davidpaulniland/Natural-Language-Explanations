import statistics
import tensorflow as tf
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance

from sklearn.inspection import permutation_importance
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence 
from sklearn.inspection import PartialDependenceDisplay

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from moepy import lowess, eda

from alibi.explainers import ALE, plot_ale
from itertools import tee, islice, chain
