from typing import List
import itertools as it
from random import choices
from collections import Counter
from typing import Union

import numpy as np
from numpy.typing import NDArray as Arr
import networkx as nx
import matplotlib.pyplot as plt

# Матрица варианта 174
P = [
        [0.24,     0,  0.17,     0,     0,     0,  0.08,     0,  0.16,     0,     0,     0,  0.35,     0,     0],
        [   0,  0.29,     0,     0,     0,     0,     0,     0,     0,     0,  0.25,     0,   0.2,     0,  0.26],
        [0.16,     0,   0.1,     0,     0,     0,  0.13,     0,  0.13,  0.48,     0,     0,     0,     0,     0],
        [   0,     0,     0,  0.12,  0.52,     0,     0,  0.36,     0,     0,     0,     0,     0,     0,     0],
        [   0,     0,     0,  0.29,  0.34,     0,     0,  0.37,     0,     0,     0,     0,     0,     0,     0],
        [   0,     0,     0,     0,     0,  0.12,     0,     0,     0,  0.52,     0,  0.24,     0,  0.12,     0],
        [0.08,     0,  0.22,  0.39,     0,     0,  0.13,     0,  0.18,     0,     0,     0,     0,     0,     0],
        [   0,     0,     0,  0.12,  0.76,     0,     0,  0.12,     0,     0,     0,     0,     0,     0,     0],
        [0.24,     0,  0.23,     0,     0,     0,   0.3,     0,  0.23,     0,     0,     0,     0,     0,     0],
        [   0,     0,     0,     0,     0,   0.2,     0,     0,     0,   0.3,     0,  0.21,     0,  0.29,     0],
        [   0,  0.76,     0,     0,     0,     0,     0,     0,     0,     0,  0.12,     0,  0.12,     0,     0],
        [   0,     0,     0,     0,     0,     0,     0,     0,     0,  0.52,     0,  0.24,     0,  0.24,     0],
        [   0,   0.4,     0,     0,     0,     0,     0,     0,     0,     0,  0.36,     0,  0.12,     0,  0.12],
        [   0,     0,     0,     0,     0,  0.36,     0,     0,     0,  0.52,     0,     0,     0,  0.12,     0],
        [   0,   0.4,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  0.48,     0,  0.12]
]


def draw_labeled_multigraph(G, attr_name, ax=None):
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", connectionstyle=connectionstyle, ax=ax, width=1, node_size=500
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
            if matrix[i][j] != 0:
                graph.add_edge(node_names[i], node_names[j], p=matrix[i][j])
    draw_labeled_multigraph(graph, "p")
    plt.show()


# Если не смогли вернуться в вершину, то тогда состояние несущественное
# Если смогли, то все пройденные относятся к одному классу существенных
# После нахождения класса другие его вершины можно не рассматривать
def significance(matrix: List[List[float]]) -> List[List[Arr]]:
    ln = len(matrix)
    seen = dict()

    def dfs(cur: int) -> Union[bool, int]:
        next_nodes = {ind for ind in range(ln) if matrix[cur][ind] != 0}
        for node in next_nodes:
            if node in seen:
                seen[cur] = seen[cur] or seen[node]
            else:
                seen[node] = False
                seen[cur] = dfs(node)
                if seen[cur] == -1:
                    return -1
        if seen[cur] is False:
            return -1
        return True

    classes = list()
    significant = set()
    for condition in range(ln):
        if not(condition in significant):
            seen[condition] = True
            res = dfs(condition)
            if res is True:
                classes.append(list(seen.keys()))
                significant = significant | seen.keys()
            seen = dict()
    print("Классы существенных состояний:")
    print(classes, '\n')
    return classes


def switch_nodes(matrix: List[List[float]], classes: List[List[int]]) -> List[List[int]]:
    permutation = list()
    seen = set()
    for clss in classes:
        for node in clss:
            permutation.append(node)
            seen.add(node)
    for node in {i for i in range(len(matrix))} - seen:
        permutation.append(node)
    new_matrix = np.array(matrix)
    new_matrix = new_matrix[permutation]
    new_matrix = new_matrix[:, permutation]
    return new_matrix


def lim_p(matrix: List[List[float]], class_ind: int) -> Arr[Arr[float]]:
    ln = len(matrix)
    p_matrix = np.array([np.array(matrix[i]) for i in range(ln)])  # Матрица P
    pt_matrix = np.transpose(p_matrix)
    e_matrix = np.eye(ln)  # Единичная матрица
    p_matrix = pt_matrix - e_matrix  # Вычитание единичной матрицы
    for i in range(ln):
        p_matrix[-1][i] = 1  # Замена последней строки

    right_part = [0 for i in range(ln)]
    right_part[-1] = 1
    print(right_part, p_matrix)
    right_part = np.array(right_part)

    p_vector = np.linalg.solve(p_matrix, right_part)
    # pi_matrix = np.array([p_vector for i in range(ln)])
    print(f"\n Предельный вектор переходов для класса {class_ind + 1}:")
    print(p_vector, '\n')
    return p_vector


def lim_p_for_classes(matrix: List[List[float]], classes: List[List[int]]):
    counter = 0
    ans = list()
    for i in range(len(classes)):
        ln = len(classes[i])
        local_matrix = [[matrix[j + counter][k + counter] for k in range(ln)] for j in range(ln)]
        counter += ln
        ans.append(lim_p(local_matrix, i))
    return ans


def global_pi(matrix: List[List[float]], classes: List[List[float]]) -> List[List[float]]:
    sign_count = sum([len(i) for i in classes])
    classes_count = len(classes)
    ln = len(matrix)

    short_ln = classes_count + ln - sign_count
    short_matrix = [[0. for i in range(short_ln)] for j in range(short_ln)]
    for i in range(sign_count, ln):
        for j in range(sign_count, ln):
            short_matrix[i - sign_count][j - sign_count] = matrix[i][j]
    for i in range(ln - sign_count, short_ln):
        short_matrix[i][i] = 1
    for i in range(sign_count, ln):
        for j in range(sign_count):
            if matrix[i][j] != 0:
                sm = 0
                for k in range(len(classes)):
                    if j >= sm:
                        clss = k
                    sm += len(classes[k])
                short_matrix[i - sign_count][ln - sign_count + clss] = matrix[i][j]
    short_matrix = np.array(short_matrix)

    print('Сокращённая матрица: \n', short_matrix, '\n')
    short_matrix_pi = np.linalg.matrix_power(short_matrix, 100)
    print('Предельная сокращённая матрица \n', short_matrix_pi, '\n')

    global_matrix = [[0. for i in range(ln)] for j in range(ln)]
    sm = 0
    for i in range(len(classes)):
        clss_ln = len(classes[i])
        for j in range(clss_ln):
            for k in range(clss_ln):
                global_matrix[j + sm][k + sm] = classes[i][k]
        sm += clss_ln
    print("\nСобранная глобальная матрица:")

    for i in range(sign_count, ln):
        sm = 0
        for j in range(classes_count):
            clss_ln = len(classes[j])
            for k in range(clss_ln):
                global_matrix[i][k + sm] = classes[j][k] * short_matrix_pi[i - sign_count][j + (ln - sign_count)]
            sm += clss_ln
    for j in global_matrix:
        print(j)
    return global_matrix


def imitation(matrix: List[List[float]], switches: int, start_cond: int) -> List[int]:
    ln = len(matrix)
    conditions = [i for i in range(ln)]
    history = [0 for i in range(switches + 1)]
    s = conditions[start_cond]
    history[0] = s
    for i in range(switches):
        s = choices(conditions, matrix[s])[0]
        history[i + 1] = s
    return history


def iterate_imitation(matrix: List[List[float]], switches: int, classes: List[List[float]]):
    result = list()
    ln = len(matrix)
    # ent = list()

    significant = sum([len(i) for i in classes])
    not_sign_counter = 0
    sign_counter = 0
    for i in range(ln):
        for k in range(10):
            cur = imitation(matrix, switches, i)
            result.append(cur[-1])
            if i <= significant and sign_counter < 2:
                sign_counter += 1
                plt.plot([j for j in range(switches + 1)], cur)
                plt.xlabel('step')
                plt.ylabel('S')
                plt.title("Существенное состояние")
                plt.grid()
                plt.show()
            elif i > significant and not_sign_counter < 6:
                not_sign_counter += 1
                plt.plot([j for j in range(switches + 1)], cur)
                plt.xlabel('step')
                plt.ylabel('S')
                plt.title("Несущественное состояние")
                plt.grid()
                plt.show()
            # cur = Counter(cur)
            # cur_ent = [0 for j in range(ln)]
            # for key in cur.keys():
            #     cur_ent[key] = cur[key] / switches
            # ent.append(cur_ent)

    # print("Экспериментальный вектор p: ", p)

    # ent = np.transpose(ent)
    # mean = [np.mean(ent[i]) for i in range(ln)]
    # print("Выборочные средние: ", mean)

    # result = Counter(result)
    # p = [0 for i in range(ln)]
    # for key in result.keys():
    #     p[key] = result[key] / (ln * 10)


sign_classes = significance(P)
draw_graph(P)
P = switch_nodes(P, sign_classes)
print(P)
draw_graph(P)
lim_p_classes = lim_p_for_classes(P, sign_classes)
PI = global_pi(P, lim_p_classes)
iterate_imitation(P, 100, sign_classes)





























