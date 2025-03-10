from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1

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

# Task 2


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
        
# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    """
    def __init__(self, criterion : str = "gini", 
                 max_depth : Optional[int] = None, 
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion=gini
        if criterion=="entropy":
            self.criterion = entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf 
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.X=X
        self.y=y
        self.root=self.build(X,y,depth=0)


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
    
        for dim in range(X.shape[1]):
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
           
            cumsum = onehot.cumsum(axis=0)  # shape: (n_samples, n_classes)
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
            gain_candidates[~valid] = -1  # невалидные кандидаты отвергаем
    
          
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
    

    
    def predict_proba(self, X: np.ndarray) ->  List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """
        list_dict_prob=[]
        for x in X:
            node=self.root
            while isinstance(node,DecisionTreeNode):
                if x[node.split_dim]>node.split_value:
                    node=node.right
                else:
                    node=node.left
            list_dict_prob.append(node.prob_dict)
        return list_dict_prob
    
    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 4
task4_dtc = None

task4_dtc = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, criterion="gini")