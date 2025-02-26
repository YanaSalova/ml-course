import numpy as np
import copy
import cvxopt
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

        y = y.astype(float)
        n_samples, n_features = X.shape
        K = np.dot(X, X.T) 
        Q = (y[:, None] * y[None, :]) * K
        P = matrix(Q, tc='d')
        q = matrix(-1.0 * np.ones(n_samples), tc='d')
        G_std = np.eye(n_samples) * -1.0  
        G_slack = np.eye(n_samples)       
        G = np.vstack((G_std, G_slack))
        h_std = np.zeros(n_samples)       
        h_slack = np.ones(n_samples) * self.C
        h = np.concatenate((h_std, h_slack))
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(np.zeros(1), tc='d')

   
        cvxopt.solvers.options['show_progress'] = False  
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])
        self.w = ((alphas * y)[:, None] * X).sum(axis=0)
        eps = 1e-7
        self.support = np.where(alphas > eps)[0]
        support_non_sat = np.where((alphas > eps) & (alphas < self.C - eps))[0]
        if len(support_non_sat) > 0:
            self.b = np.mean([y[i] - np.dot(self.w, X[i]) for i in support_non_sat])
        else:
            self.b = np.mean([y[i] - np.dot(self.w, X[i]) for i in self.support])

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
        return X.dot(self.w) + self.b

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
    """
    Возвращает полиномиальное ядро с заданной константой и степенью.
    """
    def kernel(X1, X2):
        return (X1 @ X2.T + c) ** power 
    return kernel

def get_gaussian_kernel(sigma=1.):
    """
    Возвращает ядро Гаусса с заданным коэффицинтом сигма
    """
    def kernel(x,y):
        return np.exp(-sigma * np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1) ** 2)
    
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
        self.alpha = None
        self.b = None
        self.X_s = None
        self.y = None

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
        eps = 1e-5
        self.support=np.where(alpha>eps)[0]
        
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