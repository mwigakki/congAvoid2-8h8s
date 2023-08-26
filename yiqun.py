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
next_switch_table = [   # 交换机编号0~7 而不用1~8，简化后面的操作
    [-1, 2, 2, 2, 2, 7, 7, 7],
    [3, -1, 3, 3, 4, 4, 6, 6],
    [0, 3, -1, 3, 5, 5, 3, 7],
    [2, 1, 2, -1, 4, 4, 6, 6],
    [5, 1, 5, 3, -1, 5, 6, 6],
    [2, 4, 2, 2, 4, -1, 7, 7],
    [7, 1, 3, 3, 4, 7, -1, 7],
    [0, 6, 2, 6, 6, 5, 6, -1]]

# 各个交换机去往各个终端流量矩阵
# 行：交换机s0,s1 ...， 列：终端h0, h1 ...
s2h = [
    [0, 345, 213, 343, 833, 383, 738, 938],
    [534, 0, 874, 453, 738, 539, 937, 327],
    [151, 416, 0, 164, 646, 466, 497, 678],
    [346, 246, 379, 0, 579, 546, 767, 647],
    [646, 216, 667, 479, 0, 620, 778, 706],
    [309, 530, 393, 343, 538, 0, 887, 811],
    [820, 299, 402, 454, 554, 678, 0, 978],
    [197, 779, 353, 567, 464, 466, 797, 0]
]
# 每个交换机的邻接交换机
switch_neighbor = [(2, 7), (3, 4, 6), (0, 3, 5, 7), (1, 2, 4, 6), (1, 3, 5, 6), (2, 4, 7), (1, 3, 4, 7), (0, 2, 5, 6)]

# 定义利用率差值、最大利用率，获取传入拓扑信息
difference_value = 0.2
max_value = 0.8     # 阈值
num_switches = len(s2h)
num_hosts = len(s2h[0])
num_traffic = num_switches * num_hosts  # 总共的流的数量

NUM_ITERATIONS = 100
NUM_ANTS = 5 # num_traffic * 1    # 蚂蚁的数量，每轮派出 num_ants 只蚂蚁，即查找树的第一层宽度为num_ants
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

pheromone = [[[0 for _ in range(num_switches)] for _ in range(num_hosts)] for _ in range(num_switches)]
pheromone = np.array(pheromone)
coord = [(i, j, k) for i in range(num_switches) for j in range(num_hosts) for k in range(num_switches)]
# coord ：坐标，(i, j, k) 表示从交换机Si发出到终端Hj的流SiHj，下一跳转发去Sk的 坐标

for x in range(num_switches):
    for y in range(num_hosts):
        for z in range(num_switches):
            if x != y and z in switch_neighbor[x]:    # x == y表示与目的终端直连的情况，x == z表示下一跳是自己的情况
                pheromone[x][y][z] = 1

probabilities = pheromone.copy()  # 概率最开始和信息素是一样的

pheromone = pheromone.astype(np.float32)
probabilities = probabilities.astype(np.float32)
# 定义蚁群算法的主函数
def ant_colony_optimization():
    global pheromone
    global num_ants
    global probabilities
    # 使用概率probabilities选择的时候需要使用 probabilities.flatten() 拉成一维
    # 概率会在每次调度计算时都更新自身，而信息素是每0个iter计算更新一次
    best_fitness = 0    # 记录最大的适应度
    num = 1
    for iter in range(NUM_ITERATIONS):
        # 记录某个位置上的概率有没有被选过
        fitness_list = []   # 适应度的列表，用来计算更新信息素
        solution_list = []   # solution的列表，里面保存了每个fitness对应的路由表，用来计算更新信息素
        # 保存p0，给之后的p_hat赋值，这样可以保证每个iter中，每个蚂蚁出发时的概率是一致的，不会存在前一个蚂蚁改变了概率，影响后一个
        p0 = probabilities.copy()
        s2s_matrix = cal_s2s_matrix(s2h)    # 一直在被更新更计算的s2s
        for i in range(NUM_ANTS):
            route_table = next_switch_table.copy()  # 一直在被计算被更新的路由表
            # 一只蚂蚁的操作 ，根据信息素计算概率
            count = 0 # count 表示每只蚂蚁走的距离，使用MAX_STEP 进行限制
            p_hat = p0.copy()    # p_hat 用来在计算中使被选中的概率置为0，这样减少了重复选择的次数
            # 还要将p_hat修剪一下，最初的路由表是不可选的
            p_hat = trim_origin_rt(p_hat)
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
                route_table[src_sw][dst_host] = next_sw     # 修改流表
                # 修改概率
                a = ((num - 1) * probabilities[src_sw][dst_host][next_sw] + pheromone[src_sw][dst_host][next_sw] ** alpha * ((1 / hop) ** beta)) / num

                # print(a, (num - 1) * probabilities[src_sw][dst_host][next_sw], pheromone[src_sw][dst_host][next_sw] ** alpha * ((1 / hop) ** beta))
                probabilities[src_sw][dst_host][next_sw] = a
                # 根据随机选出的调度策略，重新计算s2s 流矩阵，
                s2s_matrix = update_s2s_matrix(s2s_matrix, src_sw, dst_host, next_sw)
                # 计算适应度 fitness
                fitness = cal_fitness(s2s_matrix, e=0.2, g=0.8)
                fitness_list.append(fitness)
                solution_list.append((src_sw, dst_host, next_sw))    # src_sw, dst_host, next_sw 只保存这三元组，
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = (src_sw, dst_host, next_sw)
                    best_probabilities = probabilities.copy()
                count += 1
        update_pheromone(solution_list, fitness_list)
    return best_solution, best_fitness, best_probabilities

# 根据s2h矩阵算出 s2s 以供之后算链路利用率和适应度
def cal_s2s_matrix(s2h_matrix):
    s2s = [[0 for _ in range(num_switches)] for _ in range(num_switches)]
    for cur_sw, row in enumerate(next_switch_table):
        for col, connected_sw in enumerate(row):
            if connected_sw != -1:
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

# 根据s2h矩阵算出 s2s 以供之后算链路利用率和适应度
# 根据传入的策略更新s2s，就不需要每次都重新全部计算了。
def update_s2s_matrix(s2s_matrix, src_sw, dst_host, next_sw):
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
    while cur_sw not in sw_set: # 如果sw_set 中找到 cur_sw ，说明出现环路
        sw_set.add(cur_sw)
        cur_sw = route_table[cur_sw][dst_host]
        count += 1
        if cur_sw == -1:
            return count
    return -1   # 返回-1表示环路了

# 定义更新信息素的函数
def update_pheromone(solution_list, fitness_list):
    for i in range(len(solution_list)):
        for j, solution in enumerate(solution_list[:i+1]):
            src_sw, dst_host, next_sw = solution
            pheromone[src_sw][dst_host][next_sw] *= (1 - rho)    # 信息素挥发
            pheromone[src_sw][dst_host][next_sw] += rho * Q / fitness_list[j]    # 信息素增加

# 定义计算适应度的函数 适应度其实就是 MLU
def cal_fitness(s2s_matrix, e, g):
    # 可以给每条链路设置不同的带宽
    max_utilization = max(max(s2s_matrix)) / 1000
    min_utilization = max(min(s2s_matrix)) / 1000
    # 这里为什么这样设置
    # if max_utilization > g:
    #     return 0
    # if max_utilization - min_utilization > e:
    #     return 0
    return max_utilization

def trim_origin_rt(p_hat):
    # 根据最初的路由表使相应的转发策略是不可选的
    for src_sw, row in enumerate(next_switch_table):
        for dst_host, next_sw in enumerate(row):
            p_hat[src_sw][dst_host][next_sw] = 0
    return p_hat

if __name__ == "__main__":
    startTime = time.perf_counter()
    print(ant_colony_optimization())
    # 根据best_solution 修改路由表
    '''
    执行代码
    '''
    endTime = time.perf_counter()
    runTime = endTime - startTime	# 单位是秒
    print(runTime * 1000 , " ms")
