from typing import List
import itertools as it
from random import choices
from collections import Counter

import numpy as np
from numpy.typing import NDArray as Arr
import networkx as nx
import matplotlib.pyplot as plt

# Матрица варианта 174
P = [

]


def draw_labeled_multigraph(G, attr_name, ax=None):
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", connectionstyle=connectionstyle, ax=ax, width=1, node_size=1500
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


def draw_graph(matrix: List[List[float]]):
    graph = nx.MultiDiGraph()
    ln = len(matrix)
    node_names = [''] * ln
    for i in range(ln):
        node_names[i] = "S" + str(i + 1)
        graph.add_node(node_names[i])
    for i in range(ln):
        for j in range(ln):
            graph.add_edge(node_names[i], node_names[j], p=matrix[i][j])
    draw_labeled_multigraph(graph, "p")
    plt.show()


def lim_p(matrix: List[List[float]]) -> Arr[Arr[float]]:
    ln = len(matrix)
    p_matrix = np.array([np.array(matrix[i]) for i in range(ln)])  # Матрица P
    pt_matrix = np.transpose(p_matrix)
    e_matrix = np.eye(ln)  # Единичная матрица
    p_matrix = pt_matrix - e_matrix  # Вычитание единичной матрицы
    for i in range(ln):
        p_matrix[-1][i] = 1  # Замена последней строчки

    right_part = [0 for i in range(ln)]
    right_part[-1] = 1
    right_part = np.array(right_part)
    print(right_part)

    p_vector = np.linalg.solve(p_matrix, right_part)
    pi_matrix = np.array([p_vector for i in range(ln)])
    print(p_vector)
    print("")
    print("Предельная матрица переходов:")
    print(pi_matrix)
    print("")
    return pi_matrix


draw_graph(P)
PI = lim_p(P)


def imitation(matrix: List[List[float]], switches: int) -> List[int]:
    ln = len(matrix)
    conditions = [i for i in range(ln)]
    history = [0 for i in range(switches + 1)]
    s = choices(conditions)[0]
    history[0] = s
    for i in range(switches):
        s = choices(conditions, matrix[s])[0]
        history[i + 1] = s
    return history


def iterate_imitation(experiments: int, matrix: List[List[float]], switches: int):
    result = list()
    ln = len(matrix)
    ent = list()
    for i in range(experiments):
        cur = imitation(matrix, switches)
        result.append(cur[-1])
        if i < 3:
            plt.plot([i for i in range(switches + 1)], cur)
        cur = Counter(cur)
        cur_ent = [0 for i in range(ln)]
        for key in cur.keys():
            cur_ent[key] = cur[key] / switches
        ent.append(cur_ent)

    result = Counter(result)
    p = [0 for i in range(ln)]
    for key in result.keys():
        p[key] = result[key] / experiments

    print("Экспериментальный вектор p: ", p)
    plt.xlabel('step')
    plt.ylabel('S')
    plt.grid()
    # plt.show()
    ent = np.transpose(ent)
    mean = [np.mean(ent[i]) for i in range(ln)]
    std = [np.std(ent[i]) for i in range(ln)]
    print("Выборочные средние: ", mean)
    print("Среднеквадратичные: ", std)


iterate_imitation(50, P, 100)
