import networkx as nx
import numpy as np
from networkx.algorithms.coloring import greedy_color
import matplotlib.pyplot as plt


# Матрица смежности
adj_matrix = np.array([
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])
# Убираем петли (пути в самого себя) из матрицы смежности
np.fill_diagonal(adj_matrix, 0)

# Создаем граф
G = nx.from_numpy_array(adj_matrix)

# 1. Основная информация
vertex_count = G.number_of_nodes()  # |V|
edge_count = G.number_of_edges()    # |E|
print(f"Количество вершин (|V|): {vertex_count}")
print(f"Количество занятий с несколькими группами в одной аудитории (|E|): {edge_count}")

# 2. Хроматическое число (χ(G))
coloring = greedy_color(G, strategy="largest_first")
num_colors = len(set(coloring.values()))  # χ(G)
print(f"Учебные дни (χ(G)): {num_colors}")

# 3. Максимальная степень вершин (Δ(G))
max_degree = max(dict(G.degree()).values())
print(f"Максимальная степень вершин (Δ(G)): {max_degree}")

# 4. Проверка "окон" в расписании
schedule = {}
for node, color in coloring.items():
    if color not in schedule:
        schedule[color] = []
    schedule[color].append(node)

# Проверяем, есть ли "окна" в расписании: если более 6 пар в расписании одного цвета
has_gaps = any(len(day) > 6 for day in schedule.values())
print(f"Есть ли окна в расписании: {'Да' if has_gaps else 'Нет'}")

# (f) Число независимых множеств (β(G))
independent_sets = nx.approximation.maximum_independent_set(G)
independent_set_size = len(independent_sets)
print(f"Количество вершин в максимальном независимом множестве (β(G)): {independent_set_size}")

# (g) Характеристика "независимого множества" (ϕ(G))
phi_G = max_degree + 1 - independent_set_size  # ϕ(G)
print(f"ϕ(G): {phi_G}")

# 5. Вычисляем n^2 / (n^2 - 2m)
n = vertex_count
m = edge_count
if n**2 - 2*m != 0:
    ratio = n**2 / (n**2 - 2*m)
    print(f"(n^2) / (n^2 - 2m) = {ratio:.4f}")
else:
    print("Деление на ноль невозможно для (n^2) / (n^2 - 2m).")

# Получаем раскраску графа
coloring = greedy_color(G, strategy="largest_first")

# Создаем список цветов для каждой вершины
node_colors = [coloring[node] for node in G.nodes()]

# Рисуем граф с раскраской
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # Можно использовать nx.circular_layout(G) для кругового расположения
nx.draw(G, pos, with_labels=True,
        labels={i: i + 1 for i in range(vertex_count)},  # Изменяем индексы на 1
        node_color=node_colors,  # Используем раскраску
        node_size=700,
        font_size=10,
        font_color='black',
        edge_color='gray')
plt.title("Граф с раскраской узлов")
plt.show()



