from typing import List, Set
import itertools as it
from random import choices
from random import randint, seed
from collections import deque

import numpy as np
from numpy.typing import NDArray as Arr
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def draw_labeled_multigraph(G, attr_name, ax=None):
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    pos = nx.kamada_kawai_layout(G)
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
        node_names[i] = "S" + str(i)
        graph.add_node(node_names[i])
    for i in range(ln):
        for j in range(ln):
            if matrix[i][j] != 0:
                graph.add_edge(node_names[i], node_names[j], p=matrix[i][j])
    draw_labeled_multigraph(graph, "p")
    plt.show()


def step_algorithm(matrix: List[List[float]], groups: List[int]):
    ln = len(matrix)
    graph = dict()
    s = dict()
    for i in range(ln):
        graph[i] = dict()
        s[i] = 0
        for j in range(ln):
            if matrix[i][j] != 0:
                graph[i][j] = matrix[i][j]
                s[i] += matrix[i][j]
    answer = []
    parts = []
    counter = 0
    while len(graph) > 0:
        counter += 1
        print(f"Итерация номер {counter}. Незаполненные группы: {groups}.")
        print(f"Состояние графа: {graph}")
        mn_s = min(s, key=s.get)
        print(f"S на этой итерации: {s}")
        print(f"Минимальный S на этой итерации: вершина {mn_s}, s = {s.get(mn_s)}")
        candidate = set(graph[mn_s].keys())
        candidate.add(mn_s)
        print(f"Кандидат на группу: {candidate}")
        if len(candidate) in groups:
            print(f"Есть группа подходящего размера! ({len(candidate)})")
        else:
            # nearest = min(groups, key=lambda x: abs(x - len(candidate)))  # Получаем не заполняемость некоторых групп
            nearest = min(groups)
            print(f"Ближайшая по размеру группа: {nearest}")
            if len(candidate) > nearest:
                deltas = dict()
                for i in candidate:
                    deltas[i] = s[i] * sum([graph[i][j] for j in graph[i].keys() if not(j in candidate)])
                print(f"Дельты для удаления лишних вершин: {deltas}")
                while len(candidate) != nearest:
                    max_delta = max(deltas, key=deltas.get)
                    candidate.remove(max_delta)
                    deltas.pop(max_delta)
            else:
                inc = set()
                deltas = dict()
                for i in candidate:
                    for j in graph[i].keys():
                        if not(j in candidate) and not(j in deltas):
                            inc.add(j)
                            deltas[j] = s[j] * sum([graph[j][k] for k in graph[j].keys() if not(k in candidate)])
                while len(candidate) != nearest and len(deltas) > 0:
                    mn_s = min(deltas, key=deltas.get)
                    deltas.pop(mn_s)
                    candidate.add(mn_s)
                    for j in graph[mn_s].keys():
                        if not (j in candidate) and not (j in deltas):
                            inc.add(j)
                            deltas[j] = s[j] * sum([graph[j][k] for k in graph[j].keys() if not (k in candidate)])
        if not(len(candidate) in groups):
            parts.append(candidate)
            print("Получилась не группа, а только часть, чтобы получить полную, нужно добавить другие куски групп")
        else:
            groups.remove(len(candidate))
            answer.append(candidate)
        for cand_key in candidate:
            graph.pop(cand_key)
            s.pop(cand_key)
        for key in graph.keys():
            for cand_key in candidate:
                if cand_key in graph[key]:
                    s[key] -= graph[key][cand_key]
                    graph[key].pop(cand_key)
        print(f"Итоговая группа: {[i for i in candidate]}\n")
    print(answer)
    print(parts)


def groups_layout(switch_journal: List[int], groups: List[int]) -> List[List[int]]:
    offset = 0
    ans = []
    for i in groups:
        cur = []
        for j in range(offset, i + offset):
            cur.append(switch_journal[j])
        ans.append(cur)
        offset += i
    return ans


def iterative_algorithm(matrix: List[List[float]], groups: List[List[int]]):
    switch_journal = []
    groups = sorted(groups, key=len)
    for group in groups:
        for i in group:
            switch_journal.append(i)
    groups = [len(group) for group in groups]
    print(f"Порядок групп: {groups}")
    print("")
    p = 0
    offset_p = 0
    cnt1 = 1
    cnt2 = 1
    while p < len(groups) - 1:
        print(f"Итерация {cnt1}.{cnt2}, группа {p + 1}")
        print(f"Текущие группы: {groups_layout(switch_journal, groups)}")
        offset_q = groups[p] + offset_p
        max_delta_r = 0
        max_delta_r_pair = [0, 0]
        for q in range(p + 1, len(groups)):
            for i in range(groups[p]):
                s_qp = 0
                real_i = switch_journal[i + offset_p]
                for k in range(groups[q]):
                    real_k = switch_journal[k + offset_q]
                    s_qp += matrix[real_i][real_k]
                for k in range(groups[p]):
                    real_k = switch_journal[k + offset_p]
                    s_qp -= matrix[real_i][real_k]
                for j in range(groups[q]):
                    s_pq = 0
                    real_j = switch_journal[j + offset_q]
                    for k in range(groups[q]):
                        real_k = switch_journal[k + offset_q]
                        s_pq -= matrix[real_j][real_k]
                    for k in range(groups[p]):
                        real_k = switch_journal[k + offset_p]
                        s_pq += matrix[real_j][real_k]
                    delta_r = s_qp + s_pq - 2 * matrix[real_i][real_j]
                    if delta_r > max_delta_r:
                        max_delta_r = delta_r
                        max_delta_r_pair = [i + offset_p, j + offset_q]
            offset_q += groups[q]
        if max_delta_r <= 0:
            print("Положительную перестановку совершить невозможно, переход к рассмотрению других групп")
            cnt1 += 1
            cnt2 = 1
            offset_p += groups[p]
            p += 1
        else:
            cnt2 += 1
            first = max_delta_r_pair[0]
            second = max_delta_r_pair[1]
            print(f"Пара с максимальным delta_r ({max_delta_r}): {switch_journal[first]} - {switch_journal[second]}")
            switch_journal[first], switch_journal[second] = switch_journal[second], switch_journal[first]
        print("")


if __name__ == "__main__":
    N = [4, 5, 5, 6]
    initial = [
        [0, 0, 0, 0, 0, 8, 0, 0, 0, 1, 0, 1, 0, 0, 0, 10, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 0, 5, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 9, 8, 0],
        [0, 5, 0, 0, 2, 0, 0, 7, 4, 0, 0, 0, 0, 1, 0, 0, 2, 0, 2, 0],
        [0, 0, 6, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 7, 0],
        [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 7, 0, 0, 0, 0],
        [0, 0, 5, 0, 3, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 9, 2, 0],
        [0, 1, 0, 7, 0, 0, 0, 0, 6, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0],
        [1, 0, 8, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 11, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0, 2, 0, 0, 5],
        [1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 3, 0, 0, 9],
        [0, 3, 0, 1, 0, 0, 0, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0],
        [10, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 9, 0, 3, 0, 9, 0, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 8, 2, 7, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 9, 0, 0, 0, 4, 0, 0, 0]
    ]
    # step_algorithm(initial, N)
    # draw_graph(initial)
    initial_groups = [[1, 3, 13, 7], [4, 8, 11, 14, 17], [0, 2, 6, 9, 18], [16, 10, 19, 12, 5, 15]]
    iterative_algorithm(initial, initial_groups)
    draw_graph(initial)
