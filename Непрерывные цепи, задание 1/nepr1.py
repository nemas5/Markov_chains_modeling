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
    pos = nx.shell_layout(G)
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

print(f"lam_a = {lam_a}")
print(f"lam_b = {lam_b}")
print(f"N_A = {N_A}")
print(f"N_B = {N_B}")
print(f"R_A = {R_A}")
print(f"R_B = {R_B}")


# Предполагается нагруженный резерв A и ненагруженный резерв B
# Минимум элементов для A = 1
def fill_matrix(na, nb, ra, rb, lma, lmb) -> list:
    s = dict()
    s_list = []

    def rec_graph(a, b) -> tuple:
        if not((a, b) in s):
            s_list.append((a, b))
            if a >= 1 and b >= nb:
                s[(a, b)] = [rec_graph(a - 1, b), rec_graph(a, b - 1)]
        return a, b

    ra = 0  # если резерв ненагруженный отдельно, то убрать
    rec_graph(na + ra, nb + rb)
    ln = len(s_list)
    matrix = [[0. for j in range(ln)] for i in range(ln)]
    for i in range(ln):
        cur = s_list[i]
        if cur in s:
            ja = s_list.index(s[cur][0])
            jb = s_list.index(s[cur][1])
            if cur[0] >= na:
                matrix[i][ja] += lma * na
            else:
                matrix[i][ja] += lma * cur[0]
            matrix[i][jb] += lmb * nb
    draw_graph(matrix, s_list)
    for i in range(ln):
        matrix[i][i] -= sum(matrix[i])
    print("\nМатрица Q:")
    for i in range(ln):
        print(matrix[i])
    print('\n')
    return matrix


def solve(matrix):
    ln = len(matrix)
    p0 = np.array([0. for i in range(ln)])
    p0[0] = 1.  # Начальные условия
    still_working = set()
    for i in range(ln):  # Выбор состояний, в которых система ещё работает
        if matrix[i][i] != 0:
            still_working.add(i)
    matrix = np.transpose(matrix)

    def sodu(t, p):
        p_dif = np.dot(matrix, p)
        return p_dif

    time_interval = (0, 1)
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

    # Функция надёжности
    sol_ln = len(sol.y[0])
    p_nad = [0. for i in range(sol_ln)]
    for i in still_working:
        for j in range(sol_ln):
            p_nad[j] += sol.y[i][j]
    plt.plot(sol.t, p_nad)
    plt.xlabel("t")
    plt.ylabel("P(t)")
    plt.grid()
    plt.show()

    t_sr = simpson(p_nad, sol.t)
    print("Математическое ожидание времени безотказной работы: ", t_sr)


def imitation(matrix):
    ln = len(matrix)
    s = [i for i in range(ln)]
    res = [0. for i in range(100)]
    for k in range(100):
        t_end = 1
        t = 0.
        cur_s = 0
        while t < t_end:
            t_stay = np.random.exponential((-1) / matrix[cur_s][cur_s])
            p = [matrix[cur_s][i] / (- matrix[cur_s][cur_s]) if i != cur_s else 0 for i in range(ln)]
            cur_s = choices(s, p)[0]
            t += t_stay
            if matrix[cur_s][cur_s] == 0:
                res[k] = t
                t = 2
    mean = np.mean(res)
    print("Выборочное среднее времени безотказной работы: ", mean)
    std = np.std(res)
    print("Стандартное отклонение: ", std)


Q = fill_matrix(N_A, N_B, R_A, R_B, lam_a, lam_b)
solve(Q)
imitation(Q)
