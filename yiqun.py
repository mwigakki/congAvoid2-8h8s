#!/bin/python
import time

import numpy as np
import random
import collections

############################################此处需重新定义或是传入参数####################################
# 定义拓扑联通带宽
# 行：交换机s1,s2 ...， 列：交换机s1,s2 ...
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
'''
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
'''
next_switch_table_test = [   # 交换机编号0~7 而不用1~8，简化后面的操作
    [0, 2, 2, 2, 2, 7, 7, 7],
    [3, 1, 3, 3, 4, 4, 6, 6],
    [0, 3, 2, 3, 5, 5, 3, 7],
    [2, 1, 2, 3, 4, 4, 6, 6],
    [5, 1, 5, 3, 4, 5, 6, 6],
    [2, 4, 2, 2, 4, 5, 7, 7],
    [7, 1, 3, 3, 4, 7, 6, 7],
    [0, 6, 2, 6, 6, 5, 6, 7]]

# 各个交换机去往各个终端流量矩阵
# 行：交换机s0,s1 ...， 列：终端h0, h1 ...
s2h_test = [
    [0, 154490, 31762, 30312, 34192, 31458, 28446, 44622],
    [85384, 0, 192102, 14256, 18486, 19750, 43236, 21180],
    [28440, 32366, 0, 62504, 24772, 23744, 29130, 33418],
    [26726, 23036, 26038, 0, 68332, 22176, 24644, 31130],
    [21400, 29392, 47718, 23972, 0, 73788, 19612, 24018],
    [33848, 22544, 23626, 17408, 24586, 0, 66304, 36844],
    [14394, 32082, 19700, 17966, 15974, 35970, 0, 182062],
    [77682, 24370, 27868, 24232, 25282, 20520, 24684, 0]
]
# 每个交换机的邻接交换机
switch_neighbor = [(2, 7), (3, 4, 6), (0, 3, 5, 7), (1, 2, 4, 6), (1, 3, 5, 6), (2, 4, 7), (1, 3, 4, 7), (0, 2, 5, 6)]

# 定义利用率差值、最大利用率，获取传入拓扑信息
difference_value = 0.2
max_value = 0.8     # 阈值
num_switches = 8
num_hosts = 8
num_traffic = num_switches * num_hosts  # 总共的流的数量

NUM_ITERATIONS = 20
NUM_ANTS = 10    # num_traffic * 1    # 蚂蚁的数量，每轮派出 num_ants 只蚂蚁，即查找树的第一层宽度为num_ants
MAX_STEP = 5

farthest_distance_of_ant = 5   # 一只蚂蚁最远走的距离，即一只蚂蚁最多计算次数；
alpha = 1.0     # alpha和beta表示信息素和启发式信息的权重
beta = 2.0
rho = 0.5   # rho表示信息素挥发因子，
Q = 100     # Q表示信息素增量，pheromone表示信息素矩阵

# 定义初始的信息素，维度为( 8, 8, 8 )的矩阵，表示从交换机Si发出到终端Hj的流SiHj，下一跳转发去Sk的信息素

''' 简单结构如下
[[[s0h0->s0, s0h0->s1,..., s0h0->s7], [s0h1->s0, s0h1->s1,..., s0h7->s7]], [s1h0->s0, ..., s7h7->s6, s7h7->s7]]]
'''
pheromone = [[[1 for _ in range(num_switches)] for _ in range(num_hosts)] for _ in range(num_switches)]
pheromone = np.array(pheromone)
coord = [(i, j, k) for i in range(num_switches) for j in range(num_hosts) for k in range(num_switches)]
# coord ：坐标，(i, j, k) 表示从交换机Si发出到终端Hj的流SiHj，下一跳转发去Sk的 坐标

for x in range(num_switches):
    for y in range(num_hosts):
        for z in range(num_switches):
            if x == y or y in switch_neighbor[x] or z not in switch_neighbor[x]:
                #  这三种情况不能选：
                # 1. x != y表示与目的终端直连的情况，
                # 2. y in switch_neighbor[x] 表示目的终端所在点交换机与本交换机是直连点，此时不需要额外跳
                # 3. z not in switch_neighbor[x] 保证要转发到的下一个交换机必须是与本交换机相连点
                pheromone[x][y][z] = 0

probabilities = pheromone.copy()  # 概率最开始和信息素是一样的

pheromone = pheromone.astype(np.float32)
probabilities = probabilities.astype(np.float32)

# 定义蚁群算法的主函数
def ant_colony_optimization(s2h, next_switch_table, pheromone, probabilities):
    # 使用概率probabilities进行随机选择的时候需要使用 probabilities.flatten() 拉成一维
    # 概率会在每次调度计算时都更新自身，而信息素是每个iter计算更新一次
    best_rt = next_switch_table.copy()
    best_probabilities = probabilities.copy()
    best_fitness = 0    # 记录最大的适应度
    num = 1
    s2s_matrix_0 = cal_s2s_matrix(s2h, next_switch_table)
    for iter in range(NUM_ITERATIONS):
        # 记录某个位置上的概率有没有被选过
        fitness_list = []   # 适应度的列表，用来计算更新信息素
        solution_list = []   # solution的列表，里面保存了每个fitness对应的路由表，用来计算更新信息素
        # 保存p0，给之后的p_hat赋值，这样可以保证每个iter中，每个蚂蚁出发时的概率是一致的，不会存在前一个蚂蚁改变了概率，影响后一个
        p0 = probabilities.copy()
        for i in range(NUM_ANTS):
            route_table = next_switch_table.copy()  # 一直在被计算被更新的路由表
            s2s_matrix = s2s_matrix_0.copy()
            # 一只蚂蚁的操作 ，根据信息素计算概率
            count = 0   # count 表示每只蚂蚁走的距离，使用MAX_STEP 进行限制
            p_hat = p0.copy()    # p_hat 用来在计算中使被选中的概率置为0，这样减少了重复选择的次数
            # p_hat = trim_origin_rt(p_hat, route_table) # 将与当前流表指示的下一条的概率记录删除
            while count < MAX_STEP:
                # print("iter: ", iter, ", ant: ", i, ", count: ",count)
                num += 1
                # 随机选择调度策略
                src_sw, dst_host, next_sw = random.choices(coord, weights=p_hat.flatten())[0]
                # 计算跳数，如果跳数为-1，则表示该决策会形成环路，那么此时就不能选择此策略
                ''' 直连的情况是否需要剔除，比如s2->h7的流，本来直接发给s7，是否可以改成先发往s5，再转发到s7。
                '''
                hop = cal_hop(src_sw, dst_host, next_sw, route_table)
                if hop == -1:
                    # 形成环路的策略只是对此时来说不能选，但是之后的其他策略可能使得该策略不会再成环，所以不能把此策略的概率置为0
                    continue    # 所以直接重新选一个就行

                p_hat[src_sw][dst_host][next_sw] = 0    # 已选过的策略不能再被选，就直接把概率置为0
                # 修改概率
                a = ((num - 1) * probabilities[src_sw][dst_host][next_sw] +
                     pheromone[src_sw][dst_host][next_sw] ** alpha * ((1 / hop) ** beta)) / num
                probabilities[src_sw][dst_host][next_sw] = a
                # 根据随机选出的调度策略，重新计算s2s 流矩阵，
                s2s_matrix = update_s2s_matrix(s2s_matrix, s2h, src_sw, dst_host, next_sw, route_table)
                # 修改流表
                route_table[src_sw][dst_host] = next_sw
                # 计算适应度 fitness
                fitness = cal_fitness(s2s_matrix, e=0.2, g=0.8)
                fitness_list.append(fitness)
                solution_list.append((src_sw, dst_host, next_sw))    # src_sw, dst_host, next_sw 只保存这三元组，
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_rt = route_table.copy()
                    best_probabilities = p_hat.copy()
                count += 1
        update_pheromone(solution_list, fitness_list, pheromone)
    return best_rt, best_fitness, best_probabilities

# 根据s2h矩阵算出 s2s 以供之后算链路利用率和适应度
def cal_s2s_matrix(s2h_matrix, next_switch_table):
    s2s = [[0 for _ in range(num_switches)] for _ in range(num_switches)]
    for cur_sw, row in enumerate(next_switch_table):
        for col, connected_sw in enumerate(row):
            if connected_sw != cur_sw:
                s2s[cur_sw][connected_sw] += s2h_matrix[cur_sw][col]
    # 在mininet中，一条链路两端的交换机传入的数据包组成了链路的所有流量。
    # 例如 s1<->s3 链路的流量应该是 s1传入s3的流量加上s3传给s1的流量之和
    for i, _ in enumerate(s2s):
        for j, _ in enumerate(s2s[0]):
            if i <= j:
                tmp = s2s[j][i]
                s2s[j][i] += s2s[i][j]
                s2s[i][j] += tmp
    return s2s

def get_MLU(s2h_matrix, next_switch_table):
    s2s = cal_s2s_matrix(s2h_matrix, next_switch_table)
    return np.max(np.array(s2s))

# 根据s2h矩阵算出 s2s 以供之后算链路利用率和适应度
# 根据传入的策略更新s2s，就不需要每次都重新全部计算了。
def update_s2s_matrix(s2s_matrix, s2h, src_sw, dst_host, next_sw, next_switch_table):
    cur_next_sw = next_switch_table[src_sw][dst_host]
    modifiable_flow = s2h[src_sw][cur_next_sw]
    # 被调整的链路减去，
    s2s_matrix[src_sw][cur_next_sw] -= modifiable_flow
    s2s_matrix[cur_next_sw][src_sw] -= modifiable_flow
    # 调整后的链路加上
    s2s_matrix[src_sw][next_sw] += modifiable_flow
    s2s_matrix[next_sw][src_sw] += modifiable_flow
    return s2s_matrix

def cal_hop(src_sw, dst_host, next_sw, route_table):
    # 计算src_sw到dst_host的流下一跳如果走next_sw的话，整个路径的跳数
    count = 0
    sw_set = {src_sw}
    cur_sw = next_sw  # 当前交换机id
    while cur_sw not in sw_set:     # 如果sw_set 中找到 cur_sw ，说明出现环路
        sw_set.add(cur_sw)
        cur_sw = route_table[cur_sw][dst_host]
        count += 1
        if cur_sw == dst_host:  # 说明下一条就是终端了
            return count
    return -1   # 返回-1表示环路了

# 定义更新信息素的函数
'''
def update_pheromone(solution_list, fitness_list):
    fitness_0 = np.zeros((num_switches, num_hosts, num_switches))
    for i in range(len(solution_list)):
        for j, solution in enumerate(solution_list[:i+1]):
            src_sw, dst_host, next_sw = solution
            fitness_0[src_sw][dst_host][next_sw] = fitness_list[i] * i / len(solution)
    for index, value in np.ndenumerate(pheromone):
        src_sw, dst_host, next_sw = index
        pheromone[src_sw][dst_host][next_sw] *= (1 - rho) # 信息素挥发
        pheromone[src_sw][dst_host][next_sw] += Q * fitness_0[src_sw][dst_host][next_sw] # 信息素增加
'''
# 定义更新信息素的函数
def update_pheromone(solution_list, fitness_list, pheromone):
    for i in range(len(solution_list)):
        for j, solution in enumerate(solution_list[:i+1]):
            src_sw, dst_host, next_sw = solution
            pheromone[src_sw][dst_host][next_sw] *= (1 - rho)    # 信息素挥发,这样挥发次数太多了
            pheromone[src_sw][dst_host][next_sw] += Q * fitness_list[i]    # 信息素增加

# 定义计算适应度的函数 适应度其实就是 MLU
def cal_fitness(s2s_matrix, e, g):
    # 可以给每条链路设置不同的带宽
    # 现在每条路的带宽设置为 10mb/s  约为1.2MB/S，但这里除以1200000仅当每一秒查询一次的情况
    max_utilization = max(max(s2s_matrix)) / 1200000
    min_utilization = max(min(s2s_matrix)) / 1200000
    # MLU太大，说明这个策略不行，
    if max_utilization > g:
        return 0
    if max_utilization - min_utilization > e:
        return 0
    return max_utilization

def trim_origin_rt(p_hat, next_switch_table):
    # 根据最初的路由表使相应的转发策略是不可选的
    for src_sw, row in enumerate(next_switch_table):
        for dst_host, next_sw in enumerate(row):
            p_hat[src_sw][dst_host][next_sw] = 0
    return p_hat

''''''
if __name__ == "__main__":
    startTime = time.perf_counter()
    best_rt, best_fitness, best_probabilities = ant_colony_optimization(s2h_test, next_switch_table_test, pheromone.copy(), probabilities.copy())
    # print(best_rt, best_fitness, best_probabilities )

    # 根据best_solution 修改路由表
    # 执行代码
    endTime = time.perf_counter()
    runTime = endTime - startTime	# 单位是秒
    print(runTime * 1000, " ms")
