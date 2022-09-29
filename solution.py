# Лабораторная работа. Студент: Сахарин Н.А.

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting as pp
import patsy as pt
import pylab
import quandl
import sklearn.linear_model as lm
import sklearn.model_selection as md

def challenge_model():
    data_frame = pd.read_csv('./data/challenge_dataset.txt') # Извлечем данные из файла
    data_frame = (data_frame - data_frame.mean()) / data_frame.std() # Нормализуем
    train, test = md.train_test_split(data_frame, test_size=0.2) # Поделим на две части для обучения и проверки 80% / 20%

    # Данные для обучения
    x = train.iloc[:, 0] # выберем превый столбец (с индексом 0)
    y = train.iloc[:, 1] # выберем второй столбец (с индексом 1)

    # Данные для проверки
    test_x = test.iloc[:, 0] # выберем превый столбец (с индексом 0)
    test_y = test.iloc[:, 1] # выберем второй столбец (с индексом 1)

    x_val = np.linspace(-4, 4, 1000)

    # используем формулу языка R
    dm_y, dm_x = pt.dmatrices("y ~ x", train) # предположим, что Y линейно зависит от X
    coef = np.linalg.lstsq(dm_x, dm_y) # решим линейное уравнение, найдем коэффициенты регрессии
    coef = coef[0].ravel() # превратим в непрерывный массив
    print(coef)

    plt.subplot(2, 1, 1) # нарисуем на верхний график
    plt.plot(x, y, 'go', color = 'blue') # go означает поточечно отрисовать
    f_val = coef[0] + coef[1] * x_val # строим прямую
    plt.plot(x_val, f_val, color = 'red')

    plt.subplot(2, 1, 2) # нарисуем на нижний график
    plt.plot(test_x, test_y, 'go', color = 'blue')
    f_test_val = coef[0] + coef[1] * x_val
    plt.plot(x_val, f_test_val, color = 'red') # прямая с коэффициентами получеными в результате тренировки

    dm_y, dm_x = pt.dmatrices("test_y ~ test_x", test)
    coef = np.linalg.lstsq(dm_x, dm_y)
    coef = coef[0].ravel()
    print(coef)

    f_val = coef[0] + coef[1] * x_val
    plt.plot(x_val, f_val, color = 'green') # прямая с коэфицентами полученными на тестовых данных

    plt.show()


def global_co_model():
    data_frame = pd.read_csv('./data/global_co2.csv')
    data_frame = (data_frame - data_frame.mean()) / data_frame.std()

    #pp.scatter_matrix(data_frame, alpha=0.2, figsize=(16, 9), marker ='-')

    data_frame_col = data_frame.dropna(axis=1)
    data_frame_row = data_frame.dropna(axis=0)
    data_frame_mean = data_frame.fillna(data_frame['PerCapita'].mean())

    x_val = np.linspace(-4, 4, 1000)
    i = 0
    for frame in [data_frame_col, data_frame_row, data_frame_mean]:
        x = frame.iloc[:, 0]
        y_slice = [1, -1]
        r_exp = "Total + PerCapita ~ Year"
        if frame.shape[1] != data_frame.shape[1] :
            y_slice = [1]
            r_exp = "Total ~ Year"
        y = frame.iloc[:, y_slice]
        dm_y, dm_x = pt.dmatrices(r_exp, frame)
        coef = np.linalg.lstsq(dm_x, dm_y)[0].T
        print(coef)

        for j in range(len(y_slice)) :
            plt.subplot(2, 3, i + j * 3 + 1)
            z = y.iloc[:, j] if len(y.shape) > 1 else y
            plt.plot(x, z, 'go', color = 'blue')
            f_val = coef[j][0] + coef[j][1] * x_val
            plt.plot(x_val, f_val, color = 'red')
            plt.title(z.name)
        i += 1
    plt.show()

def google_model():
    google_dataset = quandl.get('Wiki/GOOGL').iloc[:, 0:4] # Выберем только 4 столбца из всего фрейма
    x = google_dataset.index.ravel().reshape(-1, 1).astype(int)
    x = (x - x.mean()) / x.std()
    y = google_dataset.values
    model = lm.LinearRegression()
    model.fit(x, y)
    print(model.coef_)

    x_val = np.linspace(x[0], x[-1], 1000)
    index = 0
    for elem in google_dataset.columns:
        pylab.subplot(1, len(model.coef_), index + 1)
        y_val = model.intercept_[index] + model.coef_[index] * x_val
        pylab.scatter(x, y[..., index], s=1, c='red')
        pylab.plot(x_val, y_val, 'go', color='blue')
        pylab.title(elem)
        index += 1
    plt.show()

def main():
    challenge_model()
    global_co_model()
    google_model()

if __name__ == '__main__':
    main()
