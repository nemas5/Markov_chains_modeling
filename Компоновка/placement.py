from typing import List
import itertools as it
from math import ceil, floor

import networkx as nx
import matplotlib.pyplot as plt


def draw_labeled_multigraph(G, pos, attr_name, ax=None):
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    pos = {str(i): pos[i] for i in pos.keys()}
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


def draw_graph(matrix: List[List[float]], pos: dict):
    graph = nx.MultiDiGraph()
    ln = len(matrix)
    node_names = [''] * ln
    for i in range(ln):
        node_names[i] = str(i)
        graph.add_node(node_names[i])
    for i in range(ln):
        for j in range(ln):
            if matrix[i][j] != 0:
                graph.add_edge(node_names[i], node_names[j], p=matrix[i][j])
    draw_labeled_multigraph(graph, pos, "p")
    plt.show()


def prep_q(matrix: List[List[int]], plate_dict):
    q = []  # В сумме Q * 2
    ln = len(matrix)
    for i in range(ln):
        partial_l = 0  # Средняя длинна без учёта 1/ro
        for j in range(ln):
            partial_l += matrix[i][j] * (abs(plate_dict[i][0] - plate_dict[j][0]) +
                                         abs(plate_dict[i][1] - plate_dict[j][1]))
        q.append(partial_l)
    return q


def placement(matrix: List[List[int]], x: int, y: int) -> None:
    ln = len(matrix)
    plate_dict = {0: [0, 0]}
    current_x = 0
    current_y = 0
    ro = [sum(row) for row in matrix]
    # Последовательный этап
    while len(plate_dict) != ln:
        max_k = - float('inf')
        max_k_index = 0
        for i in range(ln):
            if not(i in plate_dict):
                sm = sum([matrix[i][j] for j in plate_dict])
                k = 2 * sm - ro[i]
                if k > max_k:
                    max_k = k
                    max_k_index = i
        current_x += 1
        if current_x >= x:
            current_x = 0
            current_y += 1
        plate_dict[max_k_index] = [current_y, current_x]
    print("Результаты последовательного этапа:")
    print(plate_dict)
    q = prep_q(matrix, plate_dict)
    flag = True
    while flag:

        print(f"Значение Q: {sum(q) // 2}")

        l_max = 0
        l_max_index = 0
        for i in range(ln):
            l_current = (1 / ro[i]) * q[i]
            if l_current > l_max:
                l_max_index = i
                l_max = l_current
        print(f"Наибольшая средняя длина: {l_max_index}")
        x_c = 0
        y_c = 0
        for j in range(ln):
            if matrix[l_max_index][j] != 0:
                x_c += matrix[l_max_index][j] * abs(plate_dict[l_max_index][1] - plate_dict[j][1])
                y_c += matrix[l_max_index][j] * abs(plate_dict[l_max_index][0] - plate_dict[j][0])
        x_c /= ro[l_max_index]
        y_c /= ro[l_max_index]
        print(f"x_c: {x_c}")
        print(f"y_c: {y_c}")

        x_c = [ceil(x_c), floor(x_c)]
        y_c = [floor(y_c), ceil(y_c)]

        Q = sum(q) // 2
        draw_graph(matrix, plate_dict)
        flag = False
        for i in range(2):
            for j in range(2):
                if not flag:
                    to_switch = [k for k, v in plate_dict.items() if v == [x_c[i], y_c[j]]][0]
                    plate_dict_current = plate_dict.copy()
                    plate_dict_current[to_switch], plate_dict_current[l_max_index] = \
                        plate_dict_current[l_max_index], plate_dict_current[to_switch]
                    new_q = prep_q(matrix, plate_dict_current)
                    if sum(new_q) // 2 < Q:
                        Q = sum(new_q) // 2
                        plate_dict = plate_dict_current
                        print(f"Вершина, на которую меняем: {to_switch}")
                        q = new_q
                        flag = True
        print('\n')


if __name__ == "__main__":
    initial = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 0, 2, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 4, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 0, 2, 2, 0, 2, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 3, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 3, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 2, 0, 1, 0, 3],
        [0, 1, 0, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0],
        [0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 3, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2],
        [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 1, 0, 1, 0, 1],
        [2, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 4, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 2, 2, 1, 0, 0, 2, 1, 3, 4, 0, 0, 3, 0, 0, 0, 1, 1],
        [0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0],
        [0, 0, 4, 0, 0, 1, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 2, 0, 2, 0, 0, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 3, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    ]
    placement(initial, 4, 5)
