#!/usr/bin/env python
# coding=utf-8
# @Filename :ca.py.py
# @Time     :2023/7/16 下午4:06
# Author    :Luo Tong
import os
import sys
import time
import grpc

import sys
sys.path.append("/usr/local/lib/python3.6/site-packages")
# 引入原始p4模块
from p4.v1 import p4runtime_pb2
from p4.v1 import p4runtime_pb2_grpc

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../utils/'))
import p4runtime_lib.bmv2
import p4runtime_lib.helper

NUMBER_OF_SWITCH = 8

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

def print_matrix(matrix_name, matrix):
    print("打印：", matrix_name)
    for item in matrix:
        print(item)

def main():
    # s2h_10steps 保存前10次的s2h流量矩阵，单位是Byte/s
    s2h_10steps = [] # s2h_10steps[-1] = counter_value - last_counter_value
    # 交换机中点计数器的内容都是以Byte为单位的
    last_counter_value = [[0 for _ in range(NUMBER_OF_SWITCH)] for _ in range(NUMBER_OF_SWITCH)]
    p4info_helper = p4runtime_lib.helper.P4InfoHelper("./build/basic.p4.p4info.txt")
    counter_name = "MyIngress.pkt_counter"
    switchs = []
    # 得到8个交换机对象
    for i in range(NUMBER_OF_SWITCH):
        switch = p4runtime_lib.bmv2.Bmv2SwitchConnection(
            name='s%d'%(i+1),
            address='127.0.0.1:5005%d'%(i+1),
            device_id=i)
        # proto_dump_file='logs/s1-p4runtime-requests.txt'
        switchs.append(switch)
    # 得到8个连接交换机的存根
    client_stubs = []
    for i in range(NUMBER_OF_SWITCH):
        channel = grpc.insecure_channel(switchs[i].address)
        client_stubs.append(p4runtime_pb2_grpc.P4RuntimeStub(channel))

    while True:
        start_time = time.perf_counter()  # 计算程序用时的变量，以微秒为单位

        # 查询8个交换机的计数器
        counter_value = get_counter_value(switchs, client_stubs, p4info_helper, counter_name)
        # 与上一次计数器的值相减得到s2h的流量矩阵
        s2h_10steps.append([[i - j for i, j in zip(row_cur, row_last)] for row_cur, row_last in zip(counter_value, last_counter_value)])
        last_counter_value = counter_value  # 当前counter值赋给last_counter
        if len(s2h_10steps) >= 10:
            # TODO

            s2h_10steps.pop(0)

        print_matrix("流量矩阵s2h", s2h_10steps[-1])  # 打印s2h矩阵, 最后一位才是最新的s2h流量矩阵
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print('耗时： %f ms' % (run_time *1000))
        time.sleep(1)

if __name__ == "__main__":
    main()

