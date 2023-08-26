#!/usr/bin/env python
# coding=utf-8
# @Filename :ca.py.py
# @Time     :2023/7/16 下午4:06
# Author    :Luo Tong
import os
import subprocess
import re
import sys
import time
import grpc
from p4.v1 import p4runtime_pb2
from p4.v1 import p4runtime_pb2_grpc

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../utils/'))
import p4runtime_lib.bmv2
import p4runtime_lib.helper
from p4runtime_lib.error_utils import printGrpcError
from p4runtime_lib.switch import ShutdownAllSwitchConnections

def register_read4():
    p4info_helper = p4runtime_lib.helper.P4InfoHelper("./build/basic.p4.p4info.txt")
    s1 = p4runtime_lib.bmv2.Bmv2SwitchConnection(
        name='s1',
        address='127.0.0.1:50051',
        device_id=0,
        proto_dump_file='logs/s1-p4runtime-requests.txt')
    register_id = p4info_helper.get_registers_id("MyIngress.pkt_counter")
    # 构造一个读取请求
    req = p4runtime_pb2.ReadRequest()
    req.device_id = 0
    entity = req.entities.add()
    register_entry = entity.register_entry
    register_entry.register_id = register_id
    print(register_id)
    register_entry.index.index = 0
    # 发送请求并获取响应
    for res in s1.client_stub.Read(req):
        for entity in res.entities:
            return entity.register_entry.data

# p4runtime 明确提醒 register read is not supported yet


if __name__ == "__main__":
    start_time = time.perf_counter()  # 以微秒为单位
    register_read4()
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print('耗时： %f ms' % (run_time *1000))

