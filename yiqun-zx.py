#
import numpy as np
import random
import collections

############################################此处需重新定义或是传入参数####################################
# 定义拓扑联通带宽
# 行：交换机s1,s2 ...， 列：交换机s1,s2 ..., MB/s
bandwidth = [
    [0, 0, 10, 0, 0, 0, 0, 10],
    [4, 0, 4, 4, 5, 5, 7, 7],
    [1, 4, 0, 4, 6, 6, 4, 8],
    [3, 2, 3, 0, 5, 5, 7, 7],
    [6, 2, 6, 4, 0, 6, 7, 7],
    [3, 5, 3, 3, 5, 0, 8, 8],
    [8, 2, 4, 4, 5, 8, 0, 8],
    [1, 7, 3, 7, 7, 6, 7, 0]
]
# 交换机转发表：各个交换机去往各个终端下一跳到达的交换机编号（初始状态拓扑）
# 行：交换机s1,s2 ...， 列：终端h1, h2 ...
next_switch_table = [
    [0, 3, 3, 3, 3, 8, 8, 8],
    [4, 0, 4, 4, 5, 5, 7, 7],
    [1, 4, 0, 4, 6, 6, 4, 8],
    [3, 2, 3, 0, 5, 5, 7, 7],
    [6, 2, 6, 4, 0, 6, 7, 7],
    [3, 5, 3, 3, 5, 0, 8, 8],
    [8, 2, 4, 4, 5, 8, 0, 8],
    [1, 7, 3, 7, 7, 6, 7, 0]
]
# 各个交换机去往各个终端流量矩阵
# 行：交换机s1,s2 ...， 列：终端h1, h2 ...
# s2h
cur_flow_matrix = [
    [0, 3, 3, 3, 3, 8, 8, 8],
    [4, 0, 4, 4, 5, 5, 7, 7],
    [1, 4, 0, 4, 6, 6, 4, 8],
    [3, 2, 3, 0, 5, 5, 7, 7],
    [6, 2, 6, 4, 0, 6, 7, 7],
    [3, 5, 3, 3, 5, 0, 8, 8],
    [8, 2, 4, 4, 5, 8, 0, 8],
    [1, 7, 3, 7, 7, 6, 7, 0]
]
# 每个交换机的邻接交换机
switch_neighbor = [(3, 8), (4, 5, 7), (1, 4, 6, 8), (2, 3, 5, 7),
                   (2, 4, 6, 7), (3, 5, 8), (2, 4, 5, 8), (1, 3, 6, 7)]
# 定义利用率差值、最大利用率，获取传入拓扑信息
difference_value = 0.2
max_value = 0.8 # 阈值   最大利用率
num_switches = len(cur_flow_matrix)
num_hosts = len(cur_flow_matrix[0])
num_traffic = num_switches * num_hosts
# 获取流量矩阵坐标，后续选择流的时候用到
coords = [(i, j) for i in range(len(cur_flow_matrix)) for j in range(len(cur_flow_matrix[0]))]
# 定义蚂蚁数量、信息素参数等
# num_ants表示蚂蚁数量，蚂蚁数量=流数量*2，因为我们不仅要选择流，还需要选择调度策略
# num_iterations表示迭代次数
# alpha和beta表示信息素和启发式信息的权重
# rho表示信息素挥发因子，Q表示信息素增量，pheromone表示信息素矩阵
num_ants = num_traffic * 1.5
num_iterations = 100
alpha = 1.0
beta = 2.0
rho = 0.5
Q = 100
# 信息素矩阵是流的个数乘交换机的个数，由于拓扑的连通性，有些会一直保持零
mini_pheromone = np.full((num_switches, num_switches), 0.0001)
for i, neighbors in enumerate(switch_neighbor):
    for j in neighbors:
        mini_pheromone[i][j - 1] = mini_pheromone[j - 1][i] = 1
# 将矩阵中的每一行重复 num_hosts 次
repeated_rows = np.repeat(mini_pheromone, num_hosts, axis=0)
# 将重复的行堆叠起来，得到 pheromone 的矩阵
pheromone = np.vstack(repeated_rows)
# 生成归一化概率矩阵
sum_pheromone = np.sum(pheromone)
probabilities = pheromone / sum_pheromone


############################判断交换机转发表中是否存在环路（需要检查）#######################
# Args: matrix: 交换机转发表的矩阵，每行表示一个交换机，每列表示一个终端，
#              矩阵中的每个元素表示从该交换机到达该终端的下一跳交换机编号，
#              如果该元素的值为0，则表示该交换机已经是最终目的地。
# Returns: 如果交换机转发表中不存在环路，则返回 True，否则返回 False。21
def has_loop(matrix):
    for i in range(len(matrix)):
        next_sw = i
        visited = {k for k in range(1, 9)}
        for j in range(len(matrix[0])):
            while visited:
                if matrix[next_sw, j] == 0:
                    break
                else:
                    visited.remove(matrix[next_sw, j])
                    next_sw = matrix[next_sw, j] - 1
            print("Error: Failed to find next sw from sw {} to host {} in Loop check.".format(i, j))
            return False
    return True

    # # 计算每个节点的入度
    # in_degree = [sum([1 if matrix[i][j] != 0 else 0 for i in range(len(matrix))]) for j in range(len(matrix[0]))]
    # # 初始化队列，将所有入度为0的节点加入队列中
    # queue = collections.deque([i for i in range(len(in_degree)) if in_degree[i] == 0])
    # # 拓扑排序
    # result = []  # 用于存储排序结果
    # while queue:
    #     node = queue.popleft()
    #     result.append(node)
    #     # 遍历该节点的所有邻居节点
    #     for neighbor, has_edge in enumerate([1 if matrix[node][j] != 0 else 0 for j in range(len(matrix[node]))]):
    #         if has_edge:
    #             in_degree[neighbor] -= 1
    #             if in_degree[neighbor] == 0:
    #                 queue.append(neighbor)
    # # 判断是否存在环路
    # if len(result) == len(in_degree):
    #     return True
    # else:
    #     return False


#######################################################计算新的流量信息（需要修改）#######################
def update_s2h_flow(new_table, old_s2h):
    # 根据 new_table 将old_s2h 计算成 new_s2h new_s2s
    # 很怪
    # 收集到8个交换机的信息 data[-1] 后，就计算 s2s 的流量矩阵
    cur_row_flow = [[0], [0], [0], [0], [0], [0], [0], [0]]  # 保存每一行的流量大小,在之后的去归一化时有用
    # 收集到8个交换机的信息 data[-1] 后，就计算 s2s 的流量矩阵
    for i in range(8):
        for j in range(8):
            next_hop = new_table[i][j]
            if next_hop != 0:  # 0 就是直连
                s2s_10steps[-1][i][next_hop - 1] = s2s_10steps[-1][i][next_hop - 1] + s2h[i][j]
                cur_row_flow[i][0] = cur_row_flow[i][0] + s2h[i][j]
    return new_s2h, new_s2s


##########################生成新的路由表并判断是不是环路（需要检查）##########################
def generate_sw_table(row, col, nat_sw_table):
    # row 要调的流的s，  col 要调的流的h ； nat_sw_table： 当前计算用的路由表
    # return：根据输入 随机选出要修改后的下一跳交换机selected_node以及对应的路由表，最后
    # 获取当前下一跳交换机信息
    next_sw_value = nat_sw_table[row][col]
    # 获取选择节点的相邻节点列表
    node1_neighbors = switch_neighbor[row]
    node1_neighbors_filtered = [x for x in node1_neighbors if x != next_sw_value]
    # 生成一个包含列表元素下标的集合
    probabilities_sw = probabilities[row * col + col]
    sw_set = set(range(len(num_switches)))
    visited = sw_set
    while len(visited) > 0 and len(node1_neighbors_filtered) > 0:
        selected_node = random.choices(sw_set, weights=probabilities_sw)[0]
        visited.remove(selected_node)
        # 排除一些不可用的选择，比如当前路由，或者环路路由，或者不相连路由，直到选出一个可行的路，或者完全选出来。
        if selected_node in node1_neighbors_filtered:
            node1_neighbors_filtered.remove(selected_node)
            nat_sw_table[row][col] = selected_node
            if not has_loop(nat_sw_table):
                return nat_sw_table, selected_node
    # 如果选不出来，说明没有当前逸群路径就没有可调度点空间了
    print("Error: Failed to generate a new_sw_table")
    return None


###########################################有问题，是不是只计算了直连的（需要检查与沟通）####################
# 定义计算链路利用率的函数
def calculate_utilization(s2s):
    utilization = []
    for i in range(len(s2s) - 1):
        for j in range(len(s2s[0]) - 1):
            utilization.append(s2s[i, j] / bandwidth[i, j])
    return utilization


# 定义计算适应度的函数
# g阈值   e最大利用率
def calculate_fitness(s2s, e, g):
    utilization = calculate_utilization(s2s)
    max_utilization = max(utilization)
    min_utilization = min(utilization)
    if max_utilization > g:
        return 0
    if max_utilization - min_utilization > e:
        return 0
    return max_utilization


# 定义更新信息素的函数
def update_pheromone(list, fitness_values):
    for i, sublist in enumerate(list):
        for coordinate in sublist:
            row, col = coordinate
            pheromone[row][col] *= (1 - rho)  # 信息素挥发
            pheromone[row][col] += rho * Q / fitness_values[i]  # 信息素增加


############################################ 计算经过交换机数量（需要检查）###########################################
def hop_calculate(forwarding_table, start, end):
    # forwarding_table 修改后的表，start 要修改点交换机，end 要调的流的终端
    switch_count = 0  # 经过的交换机数量
    for i in range(len(forwarding_table)):
        if forwarding_table[start][end] != 0:
            switch_count += 1
            start = forwarding_table[start][end] - 1
        else:
            return switch_count
    print("Error: Failed to calculate hops")
    return None


# 更新对应流选择概率
def probabilities_calculate(probability):
    probabilities_flow = []
    for i in range(len(probability)):
        probabilities_flow.append(sum(probability[i]))
    probabilities_flow = np.array(probabilities_flow) / np.sum(probabilities_flow)
    return probabilities_flow


# 定义蚁群算法的主函数
def ant_colony_optimization():
    global pheromone
    best_solution = []
    best_fitness = 0
    for iteration in range(num_iterations):
        # 一次迭代初始化solutions、fitness_values
        solutions = []
        fitness_values = []
        for i in range(num_ants):
            # 一只蚂蚁的操作
            solution = []
            selected_set = set()
            nat_sw_table = next_switch_table
            update_flow_matrix = cur_flow_matrix
            while len(selected_set) < len(coords):
                probabilities_flow = probabilities_calculate(probabilities)
                selected_value = random.choices(coords, weights=probabilities_flow)[0]
                selected_set.add(selected_value)
                random_row, random_col = selected_value
                if next_switch_table[random_row][random_col] != 0:
                    # 随机选择除当前路由表的调度节点，生成新路由表
                    nat_sw_table, selected_node = generate_sw_table(random_row, random_col, nat_sw_table)
                    # 重新计算更新后的流量以及跳数
                    update_flow_matrix, new_s2s_flow_matrix = update_s2h_flow(nat_sw_table, update_flow_matrix)
                    hop = hop_calculate(nat_sw_table, random_row, random_col)
                    # 更新选择概率
                    probabilities[random_row * num_hosts + random_col][selected_node - 1] = \
                    pheromone[random_row * num_hosts + random_col][selected_node - 1] ** alpha * ((1 / hop) ** beta)
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    # 里面存储了pheromone的脚标信息
                    solution.append((random_row * num_hosts + random_col, selected_node - 1))
                    fitness = calculate_fitness(new_s2s_flow_matrix, e=difference_value, g=max_value)
                    fitness_values.append(fitness)
                    solutions.append(solution)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = nat_sw_table
        update_pheromone(solutions, fitness_values)
    return best_solution, best_fitness


# 调用蚁群算法函数求解最优解
best_solution, best_fitness = ant_colony_optimization()

# 输出最优解和适应度
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)