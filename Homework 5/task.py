import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass
    
    def backward(self, d):
        pass
        
    def update(self, alpha):
        pass
    
    
class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.
    
        Notes
        -----
        W и b инициализируются случайно.
        """
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features)
    
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        
        self.x = None
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).
    
        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.x = x  
        y = x @ self.W + self.b  
        return y
 
    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if d.ndim == 1:
            d = d.reshape(1, -1) 
        self.grad_W = self.x.T @ d
        self.grad_b = np.sum(d, axis=0)

        d_prev = d @ self.W.T
        return d_prev
        
    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W -= alpha * self.grad_W
        self.b -= alpha * self.grad_b
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
    

class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):
        self.x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
    
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.x = x  
        y = np.maximum(0, x)
        return y
        
    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if d.ndim == 1:
            d = d.reshape(1, -1) 
        d_prev = d.copy()
        d_prev[self.x <= 0] = 0  
        return d_prev
    

# Task 2

class Softmax(Module):
    def __init__(self):
        self.probs = None  

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.probs

    def backward(self, y: np.ndarray) -> np.ndarray:
        n = self.probs.shape[0]
        y_one_hot = np.zeros_like(self.probs)
        y_one_hot[np.arange(n), y] = 1
        grad = (self.probs - y_one_hot) / n
        return grad



class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        if modules[-1] != Softmax():
            modules.append(Softmax())
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи 
        (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """

        y = np.array(y)  
        n_samples = len(X)
    
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
    
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
    
            
                output = self.predict_proba(X_batch)
    
                grad = self.modules[-1].backward(y_batch)
                for layer in reversed(self.modules[:-1]):
                    grad = layer.backward(grad)
    
                for layer in self.modules[:-1]:
                    layer.update(self.alpha)

        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)
        
        """
        out = X
        for module in self.modules:
            out = module.forward(out)
        return out
        
    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Вектор предсказанных классов
        
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)
    
# Task 3

classifier_moons = MLPClassifier([Linear(2, 64),
                                  ReLU(),
                                  Linear(64, 64),
                                  ReLU(),
                                  Linear(64, 2)])  
classifier_blobs = MLPClassifier([Linear(2, 64),
                                  ReLU(),
                                  Linear(64, 64),
                                  ReLU(),
                                  Linear(64, 3)])

# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] + "model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        path = __file__[:-7] + "model.pth"  
        self.load_state_dict(torch.load(path))
        self.eval()  
    
    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        path = __file__[:-7] + "model.pth"
        torch.save(self.state_dict(), path)
        
def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    y_pred = model(X)
    loss = F.cross_entropy(y_pred, y)
    return loss