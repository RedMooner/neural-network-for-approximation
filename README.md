# neural-network-for-approximation
# Реализация нейронной сети с одним скрытым слоем

Данный код представляет реализацию нейронной сети с одним скрытым слоем для аппроксимации непрерывной функции многих переменных.

## ООП

Для улучшения структуры кода был добавлен ООП подход.

## Теорема Цыбенко

Теорема, доказанная Джорджем Цыбенко в 1989 году, которая утверждает, что искусственная нейронная сеть прямой связи с одним скрытым слоем может аппроксимировать любую непрерывную функцию многих переменных с любой точностью.

Чтобы доказать эту теорему, в коде реализуется искусственная нейронная сеть прямой связи с одним скрытым слоем, которая аппроксимирует функцию "ApproxFunc(x)", заданную в коде.

# Комментарии к коду

Этот код реализует аппроксимацию функции с помощью искусственной нейронной сети. Ниже приведен краткий комментарий к каждой части кода.

`%matplotlib` - магическая команда Jupyter Notebook для вывода графиков в ноутбуке.

`import numpy as np` - подключение библиотеки для работы с массивами и матрицами.

`import sys` - модуль системы.

`import math` - модуль математических функций.

`import matplotlib.pyplot as plt` - подключение библиотеки для построения графиков.

`def ApproxFunc(x):` - определение функции для аппроксимации.

`pcenter = 0` - центр интервала, на котором будет проводиться аппроксимация.

`prange = 10` - длина интервала, на котором будет проводиться аппроксимация.

`step = 0.1` - шаг.

`period = np.arange(pcenter-prange , pcenter+prange,step)` - создание массива из значений интервала.

`plt.plot(period,ApproxFunc(period))` - построение графика функции для аппроксимации.

`class ApproximationNN(object):` - определение класса искусственной нейронной сети.

`def __init__(self, learning_rate=0.1, input_nodes=1, hidden_nodes=50, output_nodes=1):` - конструктор класса.

`self.weights_0_1 = np.random.normal( 0.0, hidden_nodes ** -0.5, (hidden_nodes, input_nodes))` - инициализация матрицы весов между входным и скрытым слоями.

`self.weights_1_2 = np.random.normal(0.0, output_nodes ** -0.5, (output_nodes, hidden_nodes))` - инициализация матрицы весов между скрытым и выходным слоями.

`self.sigmoid_mapper = np.vectorize(self.sigmoid)` - создание векторизованной функции сигмоиды.

`self.learning_rate = np.array([learning_rate])` - инициализация скорости обучения.

`def set_lr(lr):` - функция для изменения скорости обучения.

`def sigmoid(self, x):` - функция для вычисления сигмоиды.

`def predict(self, inputs):` - функция для прогнозирования выходного значения.

`def train(self, inputs, expected_predict):` - функция для обучения нейронной сети.

`lr = 0.00005` - скорость обучения.

`myNN = ApproximationNN(learning_rate=lr, input_nodes=1, hidden_nodes=150, output_nodes=1)` - создание объекта класса нейронной сети.

`set_count = 100` - количество наборов данных для обучения.

`rand_set = np.random.random(set_count)*2*prange-prange+pcenter` - создание набора случайных данных.

`def Train():` - функция для обучения.

`def MSE(y, Y):` - функция для вычисления среднеквадратичной ошибки.

`epochs = 100` - количество эпох обучения.

`plt.ion()` - включение интерактивного режима графиков.

`for e in range(epochs):` - цикл для обучения нейронной сети.

`for tr in range(100):` - цикл для обучения на каждом наборе данных.

`plt.clf()` - очистка графика.

`plt.plot(period, ApproxFunc(period), 'blue')` - построение графика функции для аппроксимации.

`rand_set = np.random.random(set_count)*2*prange-prange+pcenter` - создание набора случайных данных.

`result_set=np.zeros(100)` - инициализация массива результатов.

`MSE=0` - инициализация среднеквадратичной ошибки.

`for x in range(0, set_count-1):` - цикл для прогнозирования на каждом наборе данных.

`result_set[x] = myNN.predict([rand_set[x]])` - прогнозирование выходного значения.

`MSE+= (ApproxFunc(rand_set[x]) - result_set[x])**2` - вычисление среднеквадратичной ошибки.

`plt.scatter(rand_set[x], result_set[x], c='red')` - построение точек на графике.

`print(rand_set[x], rand_set[x])` - вывод значений.

`plt.pause(1)` - остановка на секунду.

`myNN.set_lr(0.000001)` - изменение скорости обучения.
