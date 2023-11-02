#!/usr/bin/env python
# coding=utf-8
# @Filename :ca.py.py
# @Time     :2023/7/16 下午4:06
# Author    :Luo Tong
import os
import sys
import time
import grpc
import yiqun as yq

sys.path.append("/usr/local/lib/python3.6/site-packages")
# 引入原始p4模块
from p4.v1 import p4runtime_pb2
from p4.v1 import p4runtime_pb2_grpc

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../utils/'))
import p4runtime_lib.bmv2
import p4runtime_lib.helper

TIME_STEPS = 10
NUMBER_OF_SWITCH = 8
host_IPs = ['10.0.1.0', '10.0.2.0', '10.0.3.0', '10.0.4.0',
            '10.0.5.0', '10.0.6.0', '10.0.7.0', '10.0.8.0', ]
# 只记录了终端的mac地址，因为P4交换机流表中与终端互联时mac地址也是必需点，交换机之间点互联mac地址无关紧要
host_macs = ['08:00:00:00:01:01', '08:00:00:00:02:02',
             '08:00:00:00:03:03', '08:00:00:00:04:04',
             '08:00:00:00:05:05', '08:00:00:00:06:06',
             '08:00:00:00:07:07', '08:00:00:00:08:08', ]
# ports[i][j] 表示 交换机si到sj的出端口
ports = [
    [3, 0, 1, 0, 0, 0, 0, 4],
    [0, 1, 0, 3, 4, 0, 5, 0],
    [4, 0, 2, 1, 0, 6, 0, 5],
    [0, 5, 3, 2, 4, 0, 6, 0],
    [0, 1, 0, 2, 4, 3, 6, 0],
    [0, 0, 2, 0, 1, 5, 0, 4],
    [0, 2, 0, 1, 3, 0, 6, 4],
    [4, 0, 3, 0, 0, 2, 1, 6]
]
# next_switch_table[i][j] 表示 交换机si到hj的下一跳交换机
next_switch_table = [   # 交换机编号0~7 而不用1~8，简化后面的操作
    [0, 7, 2, 2, 2, 7, 7, 7],  # next_switch_table [0][1] 从2改成7做测试
    [3, 1, 3, 3, 4, 4, 6, 6],   # 同时 next_switch_table[i][i] = i以供ports使用
    [0, 3, 2, 3, 5, 5, 3, 7],
    [2, 1, 2, 3, 4, 4, 6, 6],
    [5, 1, 5, 3, 4, 5, 6, 6],
    [2, 4, 2, 2, 4, 5, 7, 7],
    [7, 1, 3, 3, 4, 7, 6, 7],
    [0, 6, 2, 6, 6, 5, 6, 7]]
# 原始流表，保存以防止改忘了
# next_switch_table = [   #
#     [-1, 2, 2, 2, 2, 7, 7, 7],
#     [3, -1, 3, 3, 4, 4, 6, 6],
#     [0, 3, -1, 3, 5, 5, 3, 7],
#     [2, 1, 2, -1, 4, 4, 6, 6],
#     [5, 1, 5, 3, -1, 5, 6, 6],
#     [2, 4, 2, 2, 4, -1, 7, 7],
#     [7, 1, 3, 3, 4, 7, -1, 7],
#     [0, 6, 2, 6, 6, 5, 6, -1]]

# 必须在此主控控制器上安装流表，不能通过json文件安装。不然后续修改流表时会报grpc的错
# 可能的原因：安装流表的控制器必须和修改流表点控制器是同一个
def installRT(p4info_helper, switches, route_table):
    for sw_idx in range(8):
        # 首先写入 p4中的drop的流表
        table_entry_drop = p4info_helper.buildTableEntry(
            table_name="MyIngress.ipv4_lpm",
            default_action=True,
            action_name="MyIngress.drop",
            action_params={})
        switches[sw_idx].WriteTableEntry(table_entry_drop)
        # 再写入其他到终端的流表
        for host_idx in range(8):
            table_entry = p4info_helper.buildTableEntry(
                table_name="MyIngress.ipv4_lpm",
                match_fields={
                    "hdr.ipv4.dstAddr": [host_IPs[host_idx], 24]
                },
                action_name="MyIngress.ipv4_forward",
                action_params={
                    "dstAddr": host_macs[host_idx],     # 只有交换机直连的机器需要把mac地址写正确，交换机与交换机之间互联时mac地址无关。所以MAC地址全写主机点地址就行
                    "port": ports[sw_idx][route_table[sw_idx][host_idx]]
                })  # RECONCILE_AND_COMMIT
            # 此时安装所有的初始流表
            switches[sw_idx].WriteTableEntry(table_entry)

# 根据传入之前的流表和改之后的流表对应修改
def modifyRT(p4info_helper, switches, route_table_before, route_table_after):
    for sw_idx in range(len(route_table_before)):
        for host_idx in range(len(route_table_before[0])):
            if route_table_before[sw_idx][host_idx] != route_table_after[sw_idx][host_idx] :
                table_entry = p4info_helper.buildTableEntry(
                    table_name="MyIngress.ipv4_lpm",
                    match_fields={
                        "hdr.ipv4.dstAddr": [host_IPs[host_idx], 24]
                    },
                    action_name="MyIngress.ipv4_forward",
                    action_params={
                        "dstAddr": host_macs[host_idx],
                        "port": ports[sw_idx][route_table_after[sw_idx][host_idx]]
                    })  # RECONCILE_AND_COMMIT
                # 修改流表时必须用 MODIFY
                switches[sw_idx].ModifyTableEntry(table_entry)


# client_stub 读取这一个存根对象里点寄存器值
def read_counter(client_stub, device_id, counter_id, index = None):
    request = p4runtime_pb2.ReadRequest()
    request.device_id = device_id
    entity = request.entities.add()
    counter_entry = entity.counter_entry
    if counter_id is not None:
        counter_entry.counter_id = counter_id
    else:
        counter_entry.counter_id = 0
    if index is not None:
        counter_entry.index.index = index
    for response in client_stub.Read(request):
        yield response

# 读取8个交换机里点counter 值
def get_counter_value(switchs, client_stubs, p4info_helper, counter_name):
    counter_value = [[0 for _ in range(NUMBER_OF_SWITCH)] for _ in range(NUMBER_OF_SWITCH)]
    for i in range(NUMBER_OF_SWITCH):
        j = 0
        for response in read_counter(client_stubs[i], switchs[i].device_id,
                                     p4info_helper.get_counters_id(counter_name)):
            for entity in response.entities:  # 它只有一个entity
                counter = entity.counter_entry
                counter_value[i][j] = counter.data.byte_count
                j += 1
                # print("%s %s: %d packets (%d bytes)" % (switchs[i].name, counter_name, counter.data.packet_count, counter.data.byte_count))
    return counter_value

# 打印矩阵的帮助函数
def print_matrix(matrix_name, matrix):
    print("打印矩阵：", matrix_name)
    for item in matrix:
        print(item)


def main():
    count = 1
    cur_route_table = next_switch_table  # 当前流表
    # s2h_10steps 保存前10次的s2h流量矩阵，单位是Byte/s
    s2h_10steps = [] # s2h_10steps[-1] = counter_value - last_counter_value
    # 交换机中点计数器的内容都是以Byte为单位的
    last_counter_value = [[0 for _ in range(NUMBER_OF_SWITCH)] for _ in range(NUMBER_OF_SWITCH)]
    p4info_helper = p4runtime_lib.helper.P4InfoHelper("./build/basic.p4.p4info.txt")
    counter_name = "MyIngress.pkt_counter"
    switches = []
    # 得到8个交换机对象
    for i in range(NUMBER_OF_SWITCH):
        switch = p4runtime_lib.bmv2.Bmv2SwitchConnection(
            name='s%d' % (i+1),
            address='127.0.0.1:5005%d' % (i+1),
            device_id=i,)
        # proto_dump_file='logs/s1-p4runtime-requests.txt' 这个参数就不写了，不然每次修改流表就多一条记录，后期就太长了
        switch.MasterArbitrationUpdate()  # 向交换机发送主控握手请求,设置当前控制平面为主控平面。
        # 设置接管P4流水线以及所有转发条目
        switch.SetForwardingPipelineConfig(p4info=p4info_helper.p4info, bmv2_json_file_path="./build/basic.json")
        switches.append(switch)
    # 得到8个连接交换机的存根
    client_stubs = []
    for i in range(NUMBER_OF_SWITCH):
        channel = grpc.insecure_channel(switches[i].address)
        client_stubs.append(p4runtime_pb2_grpc.P4RuntimeStub(channel))  # P4 交换机对象

    # 安装初始流表
    installRT(p4info_helper, switches, next_switch_table)

    while True:
        start_time = time.perf_counter()  # 计算程序用时的变量，以微秒为单位

        # 查询8个交换机的计数器
        counter_value = get_counter_value(switches, client_stubs, p4info_helper, counter_name)
        # 与上一次计数器的值相减得到s2h的流量矩阵
        s2h_10steps.append([[i - j for i, j in zip(row_cur, row_last)] for row_cur, row_last in zip(counter_value, last_counter_value)])
        last_counter_value = counter_value  # 当前counter值赋给last_counter

        s2h = s2h_10steps[-1]
        print("***************** 第%d次 ****************" % count)
        print_matrix("流量矩阵s2h", s2h)  # 打印s2h矩阵, 最后一位才是最新的s2h流量矩阵
        print("改流表之前MLU = ", yq.get_MLU(s2h, cur_route_table))
        if len(s2h_10steps) >= TIME_STEPS:
            best_rt, _, _ = yq.ant_colony_optimization(s2h, cur_route_table, yq.pheromone.copy(), yq.probabilities.copy())
            # 必须给pheromone和probabilities.copy()加上.copy()，不然就进去把这两个变量给改，而每一个时隙都需要使用他俩
            print_matrix("计算出的路由表", best_rt)  # 打印s2h矩阵, 最后一位才是最新的s2h流量矩阵
            modifyRT(p4info_helper, switches, route_table_before=cur_route_table, route_table_after=best_rt)
            cur_route_table = best_rt
            print("改流表之后MLU = ", yq.get_MLU(s2h, cur_route_table))
            s2h_10steps.pop(0)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print('第%d次，耗时： %f ms \n' % (count, run_time * 1000))
        count += 1
        time.sleep(2)

if __name__ == "__main__":
    main()

