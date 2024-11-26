import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
from typing import NoReturn, Tuple, List


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    df = pandas.read_csv(path_to_csv)
    df = df.sample(frac=1).reset_index(drop=True) 
    X = df.drop('label', axis=1).to_numpy()
    y = df['label'].replace(["M", "B"], [1, 0]).astype(int).to_numpy()
    return X, y
    

def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    df = pandas.read_csv(path_to_csv)
    df = df.sample(frac=1).reset_index(drop=True) 
    X = df.drop('label', axis=1).to_numpy()
    y = df['label'].to_numpy()  
    return X, y
    
    
# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    size = X.shape[0]
    split_idx = int(size * ratio)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    return X_train, y_train, X_test, y_test
 
    
# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
   
    classes = np.unique(np.concatenate((y_true, y_pred)))

    precision = []
    recall = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))



        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(precision_cls)
        recall.append(recall_cls)
                  
    precision = np.array(precision)
    recall = np.array(recall)

    accuracy = np.sum(y_pred == y_true) / len(y_true)

    return precision, recall, accuracy
    
# Task 4


class Node:
    def nearest_neighbors(self, point: np.array, k: int):
        pass

class LeafNode(Node):
    def __init__(self, X: np.array, indices: np.array):
        self.X = X
        self.indices = indices

    def nearest_neighbors(self, point: np.array, k: int) -> Tuple[np.array, np.array]:
        leaf_elements = self.X[self.indices]
        distances = np.linalg.norm(leaf_elements - point, axis=1)
        sorted_indices = np.argsort(distances)
        neighbors = self.indices[sorted_indices]
        distances = distances[sorted_indices]
        return neighbors[:k], distances[:k]

class BasicNode(Node):
    def __init__(self, left: Node, right: Node, median: float, feature_index: int):
        self.left = left
        self.right = right
        self.median = median
        self.feature_index = feature_index

    def dist_to_hyperplane(self, point: np.array) -> float:
        return (point[self.feature_index] - self.median) ** 2

    def nearest_neighbors(self, point: np.array, k: int) -> Tuple[np.array, np.array]:
        if point[self.feature_index] < self.median:
            nearest_neighbors, distances = self.left.nearest_neighbors(point, k)
            other_side = self.right
        else:
            nearest_neighbors, distances = self.right.nearest_neighbors(point, k)
            other_side = self.left

        if len(distances) < k or distances[-1] ** 2 > self.dist_to_hyperplane(point):
            other_neighbors, other_distances = other_side.nearest_neighbors(point, k)
            combined_neighbors = np.concatenate([nearest_neighbors, other_neighbors])
            combined_distances = np.concatenate([distances, other_distances])
            sorted_indices = np.argsort(combined_distances)
            combined_neighbors = combined_neighbors[sorted_indices]
            combined_distances = combined_distances[sorted_indices]
            return combined_neighbors[:k], combined_distances[:k]
        else:
            return nearest_neighbors[:k], distances[:k]

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 100):
        self.X = X
        self.leaf_size = leaf_size
        n, m = X.shape
        self.m = m
        self.root = self.build_tree(np.arange(n), 0)

    def build_tree(self, indices: np.array, depth: int = 0, max_depth: int = 20) -> Node:
        if depth >= max_depth or len(indices) <= self.leaf_size:
            return LeafNode(self.X, indices)
        else:
            data = self.X[indices]
            max_features_to_try = int(np.log2(self.m)) + 1
            for i in range(max_features_to_try):
                feature_index = (depth + i) % self.m
                feature_values = data[:, feature_index]
                median = np.median(feature_values)
                less_mask = feature_values < median
                greater_mask = feature_values >= median
                less_indices = indices[less_mask]
                greater_indices = indices[greater_mask]
                if len(less_indices) > 0 and len(greater_indices) > 0:
                    left = self.build_tree(less_indices, depth + 1, max_depth)
                    right = self.build_tree(greater_indices, depth + 1, max_depth)
                    return BasicNode(left, right, median, feature_index)
            return LeafNode(self.X, indices)
            
    def query(self, X: np.array, k: int = 1) -> List[List[int]]:
        results = []
        for point in X:
            neighbors, _ = self.root.nearest_neighbors(point, k)
            results.append(neighbors.tolist())
        return results
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """    
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
    
    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.X = X
        self.y = y
        self.kdtree = KDTree(X, leaf_size=self.leaf_size)
        self.y_train = y
        
        
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
        results = []
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        
        for point in X:
            neighbors_indices = self.kdtree.query([point], k=self.n_neighbors)[0]
            neighbors_classes = self.y_train[neighbors_indices]
            counts = np.bincount(neighbors_classes, minlength=n_classes)
            probabilities = counts / self.n_neighbors
            results.append(probabilities)
        
        return np.array(results)
     
        
        
        
    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)