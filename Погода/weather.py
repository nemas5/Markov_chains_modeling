# from typing import List
# import itertools as it
# from random import choices
# from collections import Counter
# from typing import Union

import numpy as np
# from numpy.typing import NDArray as Arr
# import networkx as nx
import matplotlib.pyplot as plt


def read_weather(file_name: str) -> tuple[dict[str, int], dict]:
    mn = +50.
    mx = -50.
    last = 0
    count = 1
    data = dict()
    discreet_temps = {i/10: 0 for i in range(-235, 268)}
    with open(file_name, 'r') as file:
        for line in file.readlines():
            cur_weather = line.split()
            if len(cur_weather) == 3:
                cur_weather[2] = float(cur_weather[2].replace(",", '.'))
            else:
                cur_weather.append(data[last])
            if (2 <= int(cur_weather[0][3:5]) <= 4) \
                    or (int(cur_weather[0][3:5]) == 5 and int(cur_weather[0][:2]) <= 10):
                if not(cur_weather[0] in data):
                    if last in data:
                        data[last] /= count
                    data[cur_weather[0]] = 0
                    count = 0
                count += 1
                data[cur_weather[0]] += cur_weather[2]
                if cur_weather[2] > mx:
                    mx = cur_weather[2]
                elif cur_weather[2] < mn:
                    mn = cur_weather[2]
                last = cur_weather[0]
                discreet_temps[round(cur_weather[2], 1)] += 1
    print("Средняя температура за каждый день с 02.2015 по 02.2025: \n", data, '\n')
    print("Минимальная температура за период с февраля по май за 10 лет:", mn, "\nМаксимальная:", mx, '\n')
    print("Распределение температур: ", discreet_temps, '\n')
    plt.bar(list(discreet_temps.keys()), list(discreet_temps.values()))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return data, discreet_temps


temps, discr = read_weather("weather.txt")


def variant1(data: dict):
    feb_decades = [0. for i in range(10)]
    may_decades = [0. for i in range(10)]
    print("\nПервый вариант:\n")
    for i in range(15, 25):
        sm_feb_dec = 0.
        sm_may_dec = 0.
        for j in range(1, 10):
            day_feb = f"0{str(j)}.02.20{str(i)}"
            day_may = f"0{str(j)}.05.20{str(i)}"
            sm_feb_dec += data[day_feb]
            sm_may_dec += data[day_may]
        sm_feb_dec += data[f"10.02.20{str(i)}"]
        sm_may_dec += data[f"10.05.20{str(i)}"]
        feb_decades[i - 15] = sm_feb_dec / 10
        may_decades[i - 15] = sm_may_dec / 10

    print("Средние температуры в первой декаде февраля за десять лет: ", feb_decades)
    print("Средние температуры в первой декаде мая за десять лет: ", may_decades)

    # Разделение отрезка температур на интервалы, каждый из которых
    # содержит примерно равное количество значений
    num_bins_feb = 5  # Пять состояний в феврале
    quantiles_feb = np.quantile(feb_decades, np.linspace(0, 1, num_bins_feb + 1))
    num_bins_may = 5  # Пять состояний в мае
    quantiles_may = np.quantile(may_decades, np.linspace(0, 1, num_bins_may + 1))
    print("")
    print("Сетка на температурах в феврале: ", quantiles_feb)
    print("Сетка на температурах в мае: ", quantiles_may)

    p_matrix = [[0 for i in range(num_bins_may)] for j in range(num_bins_feb)]

    for i in range(len(feb_decades)):
        row = 1
        col = 1
        while feb_decades[i] > quantiles_feb[row]:
            row += 1
        while may_decades[i] > quantiles_may[col]:
            col += 1
        col -= 1
        row -= 1
        p_matrix[row][col] += 1

    print("\nКоличество переходов из состояний температур в феврале в состояния температур мая:")
    for i in range(num_bins_feb):
        print(p_matrix[i])
        sm = sum(p_matrix[i])
        for j in range(num_bins_may):
            if p_matrix[i][j] > 0:
                p_matrix[i][j] /= sm
    print("\nМатрица P:")
    for i in p_matrix:
        print(i)

    sm_feb_dec = 0.
    for j in range(1, 10):
        day_feb = f"0{str(j)}.02.2025"
        sm_feb_dec += data[day_feb]
    sm_feb_dec += data[f"10.02.2025"]
    start = sm_feb_dec / 10
    print("\nСредняя температура в первой декаде февраля 2025: ", start)
    pi_vector = [0 for i in range(num_bins_feb)]
    ind = 1
    while start > quantiles_feb[ind]:
        ind += 1
    pi_vector[ind - 1] = 1
    print("\nВектор П(1): ", pi_vector)
    p_t_matrix = np.transpose(p_matrix)
    pi_vector = np.array(pi_vector)
    print("\nРезультат перемножения матрицы P и вектора П(1) - П(10): ", np.dot(p_t_matrix, pi_vector))


variant1(temps)


def variant2(data: dict, discreet_temps: dict):
    print("\nВторой вариант: \n")
    # Часть кода для второго варианта
    values_list = []
    for key, value in discreet_temps.items():
        values_list.extend([key] * value)

    num_bins = 10
    quantiles = np.quantile(values_list, np.linspace(0, 1, num_bins + 1))
    plt.hist(values_list, bins=quantiles, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()
    print(f"Сетка из {num_bins} дискретных состояний: \n", quantiles)
    tab = np.array([[0. for i in range(10)] for j in range(10)])
    for k in range(15, 25):
        for i in range(2, 6):
            sm = [0, 0, 0]
            count = [0, 0, 0]
            for j in range(31):
                if j <= 10:
                    if j == 10:
                        day = f"{str(j)}.0{str(i)}.20{str(k)}"
                    else:
                        day = f"0{str(j)}.0{str(i)}.20{str(k)}"
                    if day in data:
                        sm[0] += data[day]
                        count[0] += 1
                elif j <= 20:
                    day = f"{str(j)}.0{str(i)}.20{str(k)}"
                    if day in data:
                        sm[1] += data[day]
                        count[1] += 1
                else:
                    day = f"{str(j)}.0{str(i)}.20{str(k)}"
                    if day in data:
                        sm[2] += data[day]
                        count[2] += 1
            for j in range(3):
                if count[j] != 0 and (i - 2) * 3 + j < 10:
                    tab[k - 15][(i - 2) * 3 + j] = sm[j] / count[j]
    print("\nСредние температуры декад с первой декады февраля по первую декаду мая за 10 лет:\n", tab)
    p_matrix = [[0 for i in range(num_bins)] for j in range(num_bins)]
    for i in range(10):
        for j in range(9):
            row = 1
            col = 1
            while tab[i][j] > quantiles[row]:
                row += 1
            while tab[i][j + 1] > quantiles[col]:
                col += 1
            row -= 1
            col -= 1
            if tab[i][j] > 15:
                print(row, col)
            p_matrix[row][col] += 1
    print("\nКоличество переходов между состояниями за 10 лет:\n")
    for i in range(num_bins):
        print(p_matrix[i])
        sm = sum(p_matrix[i])
        for j in range(num_bins):
            if p_matrix[i][j] > 0:
                p_matrix[i][j] /= sm
    print("\nМатрица P: ")
    p_matrix = np.array(p_matrix)
    print(p_matrix)

    sm_feb_dec = 0.
    for j in range(1, 10):
        day_feb = f"0{str(j)}.02.2025"
        sm_feb_dec += data[day_feb]
    sm_feb_dec += data[f"10.02.2025"]
    start = sm_feb_dec / 10
    print("\nСредняя температура в первой декаде февраля 2025: ", start)
    pi_vector = [0 for i in range(num_bins)]
    ind = 1
    while start > quantiles[ind]:
        ind += 1
    pi_vector[ind - 1] = 1
    print("\nВектор П(1): ", pi_vector)
    p_t_matrix = np.transpose(p_matrix)
    p_t_matrix = np.linalg.matrix_power(p_t_matrix, 9)
    pi_vector = np.array(pi_vector)
    print("\nРезультат перемножения матрицы (P^T)^9 и вектора П(1) - П(10): ", np.dot(p_t_matrix, pi_vector))

    return tab, start


decade_temps, feb_temp = variant2(temps, discr)


def variant3(tab: list[list[float]], start: float):
    print("\nТретий вариант:\n")
    print("Температуры по декадам за десять лет:")
    print(np.transpose(tab), "\n")
    tab = np.transpose(tab)
    num_bins = 5
    quantiles = [np.quantile(tab[i], np.linspace(0, 1, num_bins + 1)) for i in range(len(tab))]
    print("Сетка температур на каждую декаду:")
    for i in quantiles:
        print(i)
    p_matrices = [[[0 for i in range(num_bins)] for j in range(num_bins)] for k in range(len(tab) - 1)]
    for k in range(len(tab) - 1):
        for i in range(len(tab[k])):
            row = 1
            col = 1
            while tab[k][i] > quantiles[k][row]:
                row += 1
            while tab[k + 1][i] > quantiles[k + 1][col]:
                col += 1
            col -= 1
            row -= 1
            p_matrices[k][row][col] += 1
    print("\nКол-во переходов между каждой парой декад:")
    for i in range(len(tab) - 1):
        print(f"{str(i + 1)} - {str(i + 2)}:")
        for j in range(num_bins):
            print(p_matrices[i][j])
            sm = sum(p_matrices[i][j])
            for k in range(num_bins):
                if p_matrices[i][j][k] > 0:
                    p_matrices[i][j][k] /= sm
    print("\nМатрицы P между каждой парой декад:")
    for i in range(len(tab) - 1):
        print(f"{str(i + 1)} - {str(i + 2)}:")
        for j in range(num_bins):
            print(p_matrices[i][j])
    print("\nСредняя температура в первой декаде февраля 2025: ", start)
    pi_vector = [0 for i in range(num_bins)]
    ind = 1
    while start > quantiles[0][ind]:
        ind += 1
    pi_vector[ind - 1] = 1
    print("\nВектор П(1): ", pi_vector)
    res = np.transpose(p_matrices[-1])
    for i in range(len(p_matrices) - 2, 0, -1):
        res = np.dot(res, np.transpose(p_matrices[i]))
    print("\nРезультат перемножения матрицы P^T(9-10) x ... x P^T(1-2) и вектора П(1) - П(10): ",
          np.dot(res, pi_vector))


variant3(decade_temps, feb_temp)
