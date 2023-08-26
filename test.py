
#!/usr/bin/env python
# coding=utf-8
# @Filename :test.py
# @Time     :2023/7/20 下午2:36
# Author    :Luo Tong

import torch
import numpy as np

if __name__ == "__main__":
    num_switches = 8

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