import numpy as np
# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return np.mean((y_true-y_predicted)**2)

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  
    ss_residual = np.sum((y_true - y_predicted) ** 2) 
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.weights
    
# Task 3

class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        """
        Инициализация модели градиентного спуска.
        
        :param alpha: Скорость обучения (learning rate).
        :param iterations: Количество итераций градиентного спуска.
        :param l: Коэффициент регуляризации Lasso.
        """
        self.alpha = alpha
        self.iterations = iterations
        self.l = l  # Коэффициент Lasso
        self.weights = None  # Место для хранения весов
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели линейной регрессии с использованием градиентного спуска.
        
        :param X: Матрица признаков (n_samples, n_features).
        :param y: Вектор целевых значений (n_samples, ).
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, n_features = X_bias.shape
        
        self.weights = np.zeros(n_features)
        
        factor = 2 / n_samples
        X_transpose = X_bias.T
        
        for iteration in range(self.iterations):
            y_pred = X_bias @ self.weights
            errors = y_pred - y
            gradient = factor * (X_transpose @ errors) + self.l * np.sign(self.weights)
            self.weights -= self.alpha * gradient

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Строим предсказания для новых данных.
        
        :param X: Матрица признаков (n_samples, n_features).
        :return: Вектор предсказаний (n_samples, ).
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.weights
# Task 4

def get_feature_importance(linear_regression):
    return np.abs(linear_regression.weights[1:])

def get_most_important_features(linear_regression):
    feature_importances = get_feature_importance(linear_regression)
    return np.argsort(-feature_importances)