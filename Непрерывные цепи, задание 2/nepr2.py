from typing import List
import itertools as it
from random import choices
from collections import Counter
from typing import Union

import numpy as np
from numpy.typing import NDArray as Arr
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simpson


def draw_labeled_multigraph(G, attr_name, ax=None):
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", connectionstyle=connectionstyle, ax=ax, width=2, node_size=1000
    )
    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )


def draw_graph(matrix: List[List[float]], nodes: list):
    graph = nx.MultiDiGraph()
    ln = len(matrix)
    node_names = [''] * ln
    for i in range(ln):
        node_names[i] = "S" + str(i + 1) + '\n' + str(nodes[i])
        graph.add_node(node_names[i])
    for i in range(ln):
        for j in range(ln):
            if matrix[i][j] != 0:
                graph.add_edge(node_names[i], node_names[j], l=matrix[i][j])
    draw_labeled_multigraph(graph, "l")
    plt.show()


N = 175
G = 6

lam_a = G + (N % 3)
lam_b = G + (N % 5)
N_A = 2 + (G % 2)
N_B = 1 + (N % 2)
R_A = 1 + (G % 2)
R_B = 2 - (G % 2)
NU = (N_A + N_B - (G % 2)) * (G + (N % 4))

print(f"lam_a = {lam_a}")
print(f"lam_b = {lam_b}")
print(f"N_A = {N_A}")
print(f"N_B = {N_B}")
print(f"R_A = {R_A}")
print(f"R_B = {R_B}")
print(f"NU = {NU}")


# Предполагается нагруженный резерв A и ненагруженный резерв B
# Минимум элементов для A = 1
def fill_matrix(na, nb, ra, rb, lma, lmb, lnu) -> tuple:
    s = dict()
    s_list = set()

    def rec_graph(a, b) -> tuple:
        if not((a, b) in s):
            if not((a, b) in s_list):
                s_list.add((a, b))
            if a > 0 and b > 0:
                s[(a, b)] = {"a": rec_graph(a - 1, b), "b": rec_graph(a, b - 1)}
            elif a == 0 and b > 0:
                s[(a, b)] = {"b": rec_graph(a, b - 1)}
            elif a > 0 and b == 0:
                s[(a, b)] = {"a": rec_graph(a - 1, b)}
            else:
                s[(a, b)] = dict()

            if 0 <= (a + b) < (na + nb + rb + ra):
                if (na + ra - a) > (nb + rb - b):
                    s[(a, b)]["nu"] = (a + 1, b)
                elif (na + ra - a) < (nb + rb - b):
                    s[(a, b)]["nu"] = (a, b + 1)
                elif lma >= lmb:
                    s[(a, b)]["nu"] = (a + 1, b)
                else:
                    s[(a, b)]["nu"] = (a, b + 1)
        return a, b

    ra = 0  # если резерв ненагруженный отдельно, то убрать
    rec_graph(na + ra, nb + rb)
    ln = len(s_list)
    s_list = list(s_list)
    start = s_list.index((na + ra, nb + rb))
    s_list[start], s_list[0] = s_list[0], s_list[start]
    matrix = [[0. for j in range(ln)] for i in range(ln)]
    for i in range(ln):
        cur = s_list[i]
        if cur in s:
            if "a" in s[cur]:
                ja = s_list.index(s[cur]["a"])
                if cur[0] >= na:
                    matrix[i][ja] += lma * na
                else:
                    matrix[i][ja] += lma * cur[0]
            if "b" in s[cur]:
                jb = s_list.index(s[cur]["b"])
                if cur[1] < nb:
                    matrix[i][jb] += lmb * cur[1]
                else:
                    matrix[i][jb] += lmb * nb
            if "nu" in s[cur]:
                jnu = s_list.index(s[cur]["nu"])
                matrix[i][jnu] += lnu

    draw_graph(matrix, s_list)
    for i in range(ln):
        matrix[i][i] -= sum(matrix[i])
    print("\nМатрица Q:")
    for i in range(ln):
        print(matrix[i])
    print('\n')
    return matrix, s_list


def kolm_algebra(matrix: List[List[float]]):
    qt_matrix = np.transpose(matrix)
    ln = len(matrix)
    for i in range(ln):
        qt_matrix[-1][i] = 1
    right_part = [0. for i in range(ln)]
    right_part[-1] = 1
    pi_vector = np.linalg.solve(qt_matrix, right_part)
    print(f"\n Предельный вектор переходов:")
    print(pi_vector, '\n')
    return pi_vector


def math_exp(pi: Arr, order: list, mn_a: int, mn_b: int):
    f = 0
    a_ready = dict()
    b_ready = dict()
    for i in range(len(order)):
        a = order[i][0]
        b = order[i][1]
        if a < mn_a or b < mn_b:
            f += pi[i]
        if not(a in a_ready):
            a_ready[a] = 0
        a_ready[a] += pi[i]
        if not(b in b_ready):
            b_ready[b] = 0
        b_ready[b] += pi[i]
    me_a = 0
    me_b = 0
    for i, j in a_ready.items():
        me_a += i * j
    for i, j in b_ready.items():
        me_b += i * j
    print("Математическое ожидание количества готовых к эксплуатации устройств типа A: ", me_a)
    print("Математическое ожидание количества готовых к эксплуатации устройств типа B: ", me_b)
    print("Вероятность отказа системы: ", f)
    print("Коэффициент загрузки ремонтной службы: ", 1 - pi[0], '\n')


def solve(matrix, pi):
    ln = len(matrix)
    p0 = np.array([0. for i in range(ln)])
    p0[0] = 1.  # Начальные условия
    matrix = np.transpose(matrix)

    def sodu(t, p):
        p_dif = np.dot(matrix, p)
        return p_dif

    time_interval = (0, 0.36936936936936937 * 2)
    t_eval = np.linspace(*time_interval, 1000)
    sol = solve_ivp(sodu, time_interval, p0, t_eval=t_eval, method='RK45')

    # Визуализация
    for i in range(len(sol.y)):
        plt.plot(sol.t, sol.y[i], label=f"P({i})")
    plt.xlabel("t")
    plt.ylabel("P(k)")
    plt.legend()
    plt.grid()
    plt.show()

    for i in range(len(sol.t)):
        pi_cur = [sol.y[j][i] for j in range(len(sol.y))]
        if np.linalg.norm(pi - pi_cur) < np.linalg.norm(pi) * 0.01:
            t_end = sol.t[i]
            print("Теоретическая оценка времени переходного процесса", sol.t[i])
            return t_end


def imitation(matrix, t_end):
    t_end *= 2
    ln = len(matrix)
    s = [i for i in range(ln)]
    t = 0.
    cur_s = 0
    res = dict()
    res[0] = 0
    while t < t_end:
        t_stay = np.random.exponential((-1) / matrix[cur_s][cur_s])
        p = [matrix[cur_s][i] / (- matrix[cur_s][cur_s]) if i != cur_s else 0 for i in range(ln)]
        cur_s = choices(s, p)[0]
        t += t_stay
        res[t] = cur_s
    print("\nМоделирование в терминах непрерывных цепей Маркова:")
    print(res)


Q, directions = fill_matrix(N_A, N_B, R_A, R_B, lam_a, lam_b, NU)
PI = kolm_algebra(Q)
math_exp(PI, directions, 1, N_B)
T = solve(Q, PI)
imitation(Q, T)
# Не уверен, что совсем правильна модель, в которой нет очереди для уже взятых на ремонт узлов
# или как-то нужно изменять коэффициент починки (вроде нет)
