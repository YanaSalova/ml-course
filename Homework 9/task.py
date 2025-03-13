from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier

# Task 0

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    pass
    
def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    pass

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    pass


# Task 1

class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        pass
        
    def predict(self, X):
        pass
    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
# Task 3

def feature_importance(rfc):
    pass

# Task 4

rfc_age = None
rfc_gender = None

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = None
catboost_rfc_gender = None