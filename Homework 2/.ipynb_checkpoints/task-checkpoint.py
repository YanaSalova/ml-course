from sklearn.neighbors import KDTree
import numpy as np
import random
import copy
from collections import deque
from typing import NoReturn
from scipy.spatial import distance_matrix

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        
        self.n_clusters=n_clusters
        self.init=init
        self.max_iter=max_iter
        self.centroids=np.zeros((n_clusters,2))
        
    def init_random(self):
        self.centroids=np.random.uniform(low=np.min(self.X), 
                                         high=np.max(self.X),
                                         size=(self.n_clusters,self.X.shape[1]))

    def init_sample(self):
        rows=np.random.choice(self.X.shape[0], self.n_clusters, replace=False)
        self.centroids=self.X[rows]
    
    def init_k_means_plusplus(self):
        random_row=np.random.randint(self.X.shape[0])
        first_centroid=self.X[random_row]
        self.centroids=np.array([first_centroid])
        dst = (np.linalg.norm(self.X - self.centroids[-1], axis=1))**2
        
        for i in range(self.n_clusters-1):
            prob=dst/np.sum(dst)
            self.centroids = np.append(self.centroids, 
                                       self.X[np.random.choice(self.X.shape[0], 1, p=prob)],
                                       axis=0)
            dst=np.minimum(dst,(np.linalg.norm(self.X - self.centroids[-1], axis=1))**2)

        
    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        self.X = X

       
        if self.init == "random":
            self.init_random()
        elif self.init == "sample":
            self.init_sample()
        elif self.init == "k-means++":
            self.init_k_means_plusplus()

        for _ in range(self.max_iter):
            labels = []
            for i in range(self.X.shape[0]):
                labels.append(np.argmin(np.linalg.norm(self.X[i] - self.centroids, axis=1)))
            labels = np.array(labels)

            for i_cluster in range(self.n_clusters):
                if np.sum(labels == i_cluster) > 0:
                    self.centroids[i_cluster] = np.mean(self.X[labels == i_cluster], axis=0)
                else:
                    self.centroids[i_cluster] = self.X[np.random.choice(self.X.shape[0])]

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        n, _ = X.shape
        distances = np.zeros((n, self.n_clusters))  
    
        
        for cluster_ind in range(self.n_clusters):
            distances[:, cluster_ind] = np.linalg.norm(X - self.centroids[cluster_ind], axis=1)
        
        labels = np.argmin(distances, axis=1)  
        return labels
# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps=eps
        self.min_samples=min_samples
        self.leaf_size=leaf_size
        self.metric=metric
        
        
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        labels=[-1]*X.shape[0]
        label_color=0
        neighbours = tree.query_radius(X, r=self.eps)
        curr_cluster = deque()
        for i in range(X.shape[0]):
            if labels[i]!=-1:
                continue
            if len(neighbours[i])<self.min_samples:
                labels[i]=0
                continue
            
            label_color+=1
            labels[i]=label_color
            curr_cluster.extend(neighbours[i])
            while curr_cluster:
                curr_dot=curr_cluster.popleft()
                if labels[curr_dot]==0:
                    labels[curr_dot]=label_color
                elif labels[curr_dot]!=-1:
                    continue
                labels[curr_dot]=label_color
                if len(neighbours[curr_dot])>=self.min_samples:
                    curr_cluster.extend(neighbours[curr_dot])
        return labels

# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        
        self.n_clusters=n_clusters
        self.linkage=linkage

    def update_dist(self,dist_matrix, clusters,upd_cluster, del_cluster, another_cluster):
        if self.linkage=="average":
            power_upd_cluster=len(clusters[upd_cluster])
            power_del_cluster=len(clusters[del_cluster])
            dist_matrix[upd_cluster][another_cluster]=(
                power_upd_cluster*dist_matrix[upd_cluster][another_cluster]+power_del_cluster*dist_matrix[del_cluster][another_cluster])/(
                    power_upd_cluster+power_del_cluster)
            dist_matrix[another_cluster][upd_cluster]=dist_matrix[upd_cluster][another_cluster]

        elif self.linkage=="single":
            dist_matrix[upd_cluster][another_cluster]=np.minimum(dist_matrix[upd_cluster][another_cluster], dist_matrix[del_cluster][another_cluster])
            dist_matrix[another_cluster][upd_cluster]=dist_matrix[upd_cluster][another_cluster]

        elif self.linkage=="complete":
            dist_matrix[upd_cluster][another_cluster]=np.maximum(dist_matrix[upd_cluster][another_cluster], dist_matrix[del_cluster][another_cluster])
            dist_matrix[another_cluster][upd_cluster]=dist_matrix[upd_cluster][another_cluster]

    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        n=X.shape[0]
        labels=np.arange(n)
        clusters=[[i] for i in range(n)]
        dist_matrix=np.empty([n,n])
        clusters_count=n

        dist_matrix=distance_matrix(np.array(X),np.array(X))        
        np.fill_diagonal(dist_matrix, np.inf)

        while clusters_count>self.n_clusters:
            min_idx=np.argmin(dist_matrix)
            upd_cluster=int(min_idx//clusters_count)
            del_cluster=int(min_idx%clusters_count)
            for i in range(clusters_count):
                if i!=upd_cluster and i!=del_cluster:
                    self.update_dist(dist_matrix, clusters, upd_cluster, del_cluster, i)

            dist_matrix=np.delete(dist_matrix,del_cluster,0)
            dist_matrix=np.delete(dist_matrix,del_cluster,1)
            labels[clusters[del_cluster]]=upd_cluster
            labels[labels > del_cluster] -= 1
            clusters[upd_cluster]+=clusters[del_cluster]
            del clusters[del_cluster]

            clusters_count-=1
        return labels