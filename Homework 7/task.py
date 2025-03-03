import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1

class LinearSVM:
    def __init__(self, C: float):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        n = X.shape[0]
        tmp = y.reshape(-1, 1) * X
        P = matrix(tmp @ tmp.T)
        q = matrix(-np.ones((n, 1)))
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), self.C * np.ones(n))))
        A = matrix(y.reshape(1, -1) * 1.)
        b = matrix(0.)
        
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x']).flatten()  
        threshold = 1e-5
        support = alpha > threshold
        self.support = np.where(support)[0]
        self.w = np.sum(alpha[support, None] * y[support, None] * X[support], axis=0)
        self.b = np.mean(y[support] - X[support] @ self.w)

    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """

        return X@self.w.T+self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))

# Task 2

def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"
    def kernel(X,Y):
        return (X@Y.T+c)**power

    return kernel

def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом sigma"
    def kernel(x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        result = np.exp(-sigma * np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)**2)
        if x.shape[0] == 1 or y.shape[0] == 1:
            result = np.squeeze(result)
        return result
    return kernel


# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.
        
        """
        self.C = C
        self.kernel = kernel
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndar/ray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """

        n=X.shape[0]
        self.y=y.reshape(-1,1)
        P=matrix(self.y@self.y.T * self.kernel(X, X))
        q=matrix(-np.ones((n,1)))
        G=matrix(np.vstack((-np.eye(n),np.eye(n))))
        h=matrix(np.hstack((np.zeros(n),self.C*np.ones(n))))
        A=matrix(y.reshape(1,-1)*1.)
        b=matrix(0.)

        sol=solvers.qp(P,q,G,h,A,b)
        alpha=np.array(sol['x']).flatten()
        threshold = 1e-5
        self.support=np.where(alpha>threshold)[0]
        
        self.alpha=alpha.reshape(-1,1)
        self.X_s = X[self.support]
        self.b = np.mean(self.y[self.support] - np.sum(self.alpha[self.support] * self.y[self.support] * self.kernel(self.X_s, self.X_s), axis=0))

        

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
    
        Xw = (self.alpha[self.support] * self.y[self.support]).T @ self.kernel(self.X_s, X)  
        return Xw.flatten() + self.b


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))