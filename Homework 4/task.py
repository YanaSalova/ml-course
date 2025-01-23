import numpy as np
import copy
from typing import NoReturn


# Task 1


class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.w = None
        self.iterations = iterations
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон.
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        unique_classes = np.unique(y)
        self.classes_ = unique_classes

        y_transformed = np.where(y == unique_classes[0], -1, 1)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.w = np.zeros(X.shape[1])

        for iteration in range(self.iterations):
            predictions = np.sign(np.dot(X, self.w))
            predictions[predictions == 0] = 1
            mask = predictions != y_transformed
            if not mask.any():
                break
            first_error_idx = np.argmax(mask)
            x_i, y_i = X[first_error_idx], y_transformed[first_error_idx]
            self.w += y_i * x_i

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        predictions = np.sign(np.dot(X, self.w))
        predictions[predictions == 0] = 1
        return np.where(predictions == -1, self.classes_[0], self.classes_[1]).astype(
            int
        )


# Task 2


class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
            Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
            Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
            w[0] должен соответствовать константе,
            w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """
        self.w = None
        self.iterations = iterations
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течение iterations итераций.

        При этом в конце обучения оставляет веса,
        при которых значение accuracy было наибольшим.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        y_transformed = np.where(y == unique_classes[0], -1, 1)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.w = np.zeros(X.shape[1])
        best_w = np.copy(self.w)
        best_accuracy = 0

        for iteration in range(self.iterations - 10000):
            predictions = np.sign(np.dot(X, self.w))
            predictions[predictions == 0] = 1
            mask = predictions != y_transformed
            self.w += np.sum(y_transformed[mask][:, None] * X[mask], axis=0)

            current_accuracy = np.mean(predictions == y_transformed)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_w = np.copy(self.w)

        self.w = best_w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор меток классов
            (по одной метке для каждого элемента из X).

        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        predictions = np.sign(np.dot(X, self.w))
        predictions[predictions == 0] = 1
        return np.where(predictions == -1, self.classes_[0], self.classes_[1]).astype(
            int
        )


def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
    Компонент 1: Средняя интенсивность пикселей изображения.
    Компонент 2: Количество пикселей с максимальной интенсивностью.

    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    n_images = images.shape[0]
    avg_intensity = images.reshape(n_images, -1).mean(axis=1)
    max_intensity = images.max(axis=(1, 2))
    count_max_intensity = (
        (images == max_intensity[:, None, None]).reshape(n_images, -1).sum(axis=1)
    )
    transformed = np.stack((avg_intensity, count_max_intensity), axis=1)
    return transformed
