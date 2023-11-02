
#!/usr/bin/env python
# coding=utf-8
# @Filename :test.py
# @Time     :2023/7/20 下午2:36
# Author    :Luo Tong

import sys
sys.path.append("/usr/local/lib/python3.6/site-packages")
# 引入原始p4模块
import grpc
from p4.v1 import p4runtime_pb2, p4runtime_pb2_grpc

# 定义交换机的IP地址和端口
switch_ip = "127.0.0.1"
switch_port = 50051
IPs = [b'\x0a\x00\x01\x00', b'\x0a\x00\x02\x00', b'\x0a\x00\x03\x00', b'\x0a\x00\x04\x00',
       b'\x0a\x00\x05\x00', b'\x0a\x00\x06\x00', b'\x0a\x00\x07\x00', b'\x0a\x00\x08\x00', ]

# 创建与交换机的gRPC通道
channel = grpc.insecure_channel(f'{switch_ip}:{switch_port}')

# 创建P4RuntimeStub
stub = p4runtime_pb2_grpc.P4RuntimeStub(channel)

# 创建P4Runtime请求消息
request = p4runtime_pb2.WriteRequest()
request.device_id = 1
request.election_id.low = 1

# 创建要修改的表项
add = request.updates.add()
add.type = p4runtime_pb2.Update.MODIFY
table_entry = add.entity.table_entry

# 设置表项的相关字段（表名、匹配字段、动作等）
# 这些表的信息都在 ./logs/sx-p4runtime-requests.txt  文件中。
table_entry.table_id = 37375156  # 表的ID
table_entry.is_default_action = False  # 是否为默认动作
# table_entry.priority = 1  # 优先级

# 添加匹配字段
match_field = table_entry.match.add()
match_field.field_id = 1  # 字段ID
match_field.lpm.value = b'\x0a\x00\x09\x00'# IPs[1]  # 要修改的目的IP（bytes）
match_field.lpm.prefix_len = 24

# 添加动作
action = table_entry.action.action
action.action_id = 28792405  # 动作ID

# 添加动作参数
action_param1 = action.params.add()
action_param1.param_id = 1  # 参数ID
action_param1.value = b'\x08\x00\x00\x00\x02\x16'  # 参数值（二进制）
action_param2 = action.params.add()
action_param2.param_id = 2  # 参数ID
action_param2.value = b'\x04'  # 参数值（二进制）

print(request)
# 发送请求到交换机
response = stub.Write(request)

