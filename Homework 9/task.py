from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 0

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    unique, counts = np.unique(x, return_counts = True)
    prob = counts/counts.sum()
    return np.sum(prob*(1-prob))
    
def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    unique, counts = np.unique(x, return_counts = True)
    prob = counts/counts.sum()
    return -np.sum(prob*np.log2(prob, where=(prob>0)))

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
    parent_y = np.concatenate((left_y, right_y))

    impurity_parent = criterion(parent_y)
    impurity_left = criterion(left_y)
    impurity_right = criterion(right_y)

  
    p_size = len(parent_y)  
    l_size = len(left_y)   
    r_size = len(right_y)  

    
    ig = impurity_parent - ((l_size / p_size) * impurity_left + (r_size / p_size) * impurity_right)
    
    return ig



# Task 1

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """
    def __init__(self, ys):
        self.y = None
        values, counts = np.unique(ys, return_counts=True)
        self.y = values[np.argmax(counts)]
        self.prob_dict=dict(zip(values, counts/len(ys)))



class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
 

class DecisionTree:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    """
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 criterion: str = "gini", 
                 max_depth: Optional[int] = None, 
                 min_samples_leaf: int = 1, 
                 max_features: Union[str, int] = "auto"):
        self.X = X
        self.y = y
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        if max_features == "auto":
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            self.max_features = int(max_features)
        self.oob_indices = None
        self.root = self.build(X, y, depth=0)

    def build(self, X, y, depth):
        if len(X) <= self.min_samples_leaf or len(np.unique(y)) == 1:
            return DecisionTreeLeaf(y)
        if self.max_depth is not None and depth == self.max_depth:
            return DecisionTreeLeaf(y)
    
        best_gain = -1
        best_dim = None
        best_value = None
        best_left_idx = None
        best_right_idx = None
        parent_impurity = self.criterion(y)
        n_samples = len(y)
    
        features_to_consider = np.random.choice(range(X.shape[1]), 
                                                size=min(self.max_features, X.shape[1]), 
                                                replace=False)
    
        for dim in features_to_consider:
            sorted_idx = np.argsort(X[:, dim])
            sorted_X = X[sorted_idx, dim]
            sorted_y = y[sorted_idx]
    
            candidate_indices = np.where(np.diff(sorted_X) != 0)[0] + 1
            if len(candidate_indices) == 0:
                continue
    
            classes, inv = np.unique(sorted_y, return_inverse=True)
            n_classes = len(classes)
    
            onehot = np.zeros((len(sorted_y), n_classes), dtype=float)
            onehot[np.arange(len(sorted_y)), inv] = 1.0
            cumsum = onehot.cumsum(axis=0)
            total_counts = cumsum[-1, :]
    
            left_counts = cumsum[candidate_indices - 1, :]
            right_counts = total_counts - left_counts
            left_n = candidate_indices.astype(float)
            right_n = n_samples - left_n
    
            left_prob = left_counts / left_n[:, None]
            right_prob = right_counts / right_n[:, None]
    
            if self.criterion == gini:
                left_impurity = 1 - np.sum(left_prob ** 2, axis=1)
                right_impurity = 1 - np.sum(right_prob ** 2, axis=1)
            else:
                left_impurity = -np.sum(np.where(left_prob > 0, left_prob * np.log2(left_prob), 0), axis=1)
                right_impurity = -np.sum(np.where(right_prob > 0, right_prob * np.log2(right_prob), 0), axis=1)
    
            weighted_impurity = (left_n / n_samples) * left_impurity + (right_n / n_samples) * right_impurity
            gain_candidates = parent_impurity - weighted_impurity
    
            valid = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
            gain_candidates[~valid] = -1
    
            idx_best = np.argmax(gain_candidates)
            if gain_candidates[idx_best] > best_gain:
                best_gain = gain_candidates[idx_best]
                best_dim = dim
                i = candidate_indices[idx_best]
                best_value = (sorted_X[i - 1] + sorted_X[i]) / 2.0
                best_left_idx = X[:, dim] < best_value
                best_right_idx = X[:, dim] >= best_value
    
        if best_gain == -1 or best_dim is None:
            return DecisionTreeLeaf(y)
    
        left_tree = self.build(X[best_left_idx], y[best_left_idx], depth + 1)
        right_tree = self.build(X[best_right_idx], y[best_right_idx], depth + 1)
        return DecisionTreeNode(best_dim, best_value, left_tree, right_tree)
    
    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        list_dict_prob = []
        for x in X:
            node = self.root
            while isinstance(node, DecisionTreeNode):
                if x[node.split_dim] > node.split_value:
                    node = node.right
                else:
                    node = node.left
            list_dict_prob.append(node.prob_dict)
        return list_dict_prob
    
    def predict(self, X: np.ndarray) -> list:
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        """
        Параметры:
            criterion : str
                Критерий для построения дерева ("gini" или "entropy").
            max_depth : Optional[int]
                Максимальная глубина дерева. Если None, то глубина не ограничена.
            min_samples_leaf : int
                Минимальное число элементов в листе.
            max_features : str или int
                Количество признаков, которые будут рассматриваться в каждом узле. Если "auto",
                то выбирается sqrt(число признаков).
            n_estimators : int
                Количество деревьев, которые будут построены.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []  

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучает Random Forest путём построения n_estimators деревьев на подвыборках (bagging).

        Параметры:
            X : np.ndarray
                Матрица признаков обучающей выборки.
            y : np.ndarray
                Вектор меток классов.
        """
        self.X = X
        self.y = y
        n_samples, n_features = X.shape

        if self.max_features == "auto":
            self.max_features_value = int(np.sqrt(n_features))
        else:
            self.max_features_value = int(self.max_features)

        self.trees = []
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTree(X_sample, y_sample,
                                criterion=self.criterion,  
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=self.max_features_value)
            tree.oob_indices = oob_indices
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Для каждого элемента выборки X возвращает итоговое предсказание (модальное предсказание по всем деревьям).
        
        Параметры:
            X : np.ndarray
                Матрица объектов для предсказания.
                
        Возвращает:
            np.ndarray с предсказанными метками классов.
        """
        predictions = []
        for x in X:
            votes = []
            for tree in self.trees:
                vote = tree.predict(np.array([x]))[0]
                votes.append(vote)
            predictions.append(max(set(votes), key=votes.count))
        return np.array(predictions)

    
# Task 3

def feature_importance(rfc):
    """
    Parameters:
        rfc: RandomForestClassifier – обученный случайный лес, в котором сохранены обучающие данные
             (rfc.X, rfc.y) и для каждого дерева сохранён атрибут oob_indices.
             
    Returns:
        np.ndarray: массив важностей признаков размерности (n_features,).
    """
    n_trees = len(rfc.trees)
    n_features = rfc.X.shape[1]
    importances = np.zeros(n_features)
    
    for tree in rfc.trees:
        oob_idx = tree.oob_indices  
        if len(oob_idx) == 0:
            continue
        X_oob = rfc.X[oob_idx]
        y_oob = rfc.y[oob_idx]
        
        y_pred = tree.predict(X_oob)
        baseline_error = np.mean(y_pred != y_oob)
        
        for j in range(n_features):
            X_perm = X_oob.copy()
            np.random.shuffle(X_perm[:, j])
            y_pred_perm = tree.predict(X_perm)
            perm_error = np.mean(y_pred_perm != y_oob)
            importances[j] += (perm_error - baseline_error)
    
    importances /= n_trees
    return importances

# Task 4

rfc_age = RandomForestClassifier(
    criterion="gini",
    max_depth=20,        
    min_samples_leaf=1,   
    max_features=5,  
    n_estimators=10   
)
rfc_gender = RandomForestClassifier(
    criterion="gini",
    max_depth=10,         
    min_samples_leaf=1,   
    max_features=5,
    n_estimators=5      
)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model


catboost_rfc_age = CatBoostClassifier(loss_function='MultiClass', random_seed=42, verbose=False)
catboost_rfc_age.load_model(__file__[:-7] +'/catboost_model_age.cbm')

catboost_rfc_gender = CatBoostClassifier(loss_function='MultiClass', random_seed=42, verbose=False)
catboost_rfc_gender.load_model(__file__[:-7] +'/catboost_model_sex.cbm')


