
import time
import os
import sys
sys.path.append("/usr/local/lib/python3.6/site-packages")
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../utils/'))
import p4runtime_lib.bmv2
import p4runtime_lib.helper

# 定义交换机的IP地址和端口
switch_ip = "127.0.0.1"
switch_port = 50051
IPs = [b'\x0a\x00\x01\x00', b'\x0a\x00\x02\x00', b'\x0a\x00\x03\x00', b'\x0a\x00\x04\x00',
       b'\x0a\x00\x05\x00', b'\x0a\x00\x06\x00', b'\x0a\x00\x07\x00', b'\x0a\x00\x08\x00', ]
host_IPs = ['10.0.1.0', '10.0.2.0', '10.0.3.0', '10.0.4.0',
            '10.0.5.0', '10.0.6.0', '10.0.7.0', '10.0.8.0', ]
# 只记录了终端的mac地址，因为P4交换机流表中与终端互联时mac地址也是必需点，交换机之间点互联mac地址无关紧要
host_macs = ['08:00:00:00:01:01', '08:00:00:00:02:02',
             '08:00:00:00:03:03', '08:00:00:00:04:04',
             '08:00:00:00:05:05', '08:00:00:00:06:06',
             '08:00:00:00:07:07', '08:00:00:00:08:08']
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

def modifyRT(p4info_helper, src_sw, next_sw , dst_host):
    table_entry = p4info_helper.buildTableEntry(
        table_name="MyIngress.ipv4_lpm",
        match_fields={
            "hdr.ipv4.dstAddr": [dst_host, 24]
        },
        action_name="MyIngress.ipv4_forward",
        action_params={
            "dstAddr": "08:00:00:00:02:22",
            "port": ports[0][next_sw]
        })  # RECONCILE_AND_COMMIT
    # 修改流表时必须用 MODIFY
    src_sw.ModifyTableEntry(table_entry)

def main():
    p4info_helper = p4runtime_lib.helper.P4InfoHelper("./build/basic.p4.p4info.txt")
    switches = []
    # 得到8个交换机对象
    for i in range(8):
        switch = p4runtime_lib.bmv2.Bmv2SwitchConnection(
            name='s%d' % (i + 1),
            address='127.0.0.1:5005%d' % (i + 1),
            device_id=i,)
        # proto_dump_file='logs/s%d-p4runtime-requests.txt' % (i + 1)
        switch.MasterArbitrationUpdate()  # 向交换机发送主控握手请求,设置当前控制平面为主控平面。
        # 设置接管P4流水线以及所有转发条目
        # switch.SetForwardingPipelineConfig2(p4info=p4info_helper.p4info, bmv2_json_file_path="./build/basic.json")
        switch.SetForwardingPipelineConfig(p4info=p4info_helper.p4info, bmv2_json_file_path="./build/basic.json")
        switches.append(switch)

    # next_switch_table[i][j] 表示 交换机si到hj的下一跳交换机
    next_switch_table = [  # 交换机编号0~7 而不用1~8，简化后面的操作
        [0, 4, 2, 2, 2, 7, 7, 7],  # next_switch_table [0][1] 从2改成7做测试
        [3, 1, 3, 3, 4, 4, 6, 6],  # 同时 next_switch_table[i][i] = i以供ports使用
        [0, 3, 2, 3, 5, 5, 3, 7],   # next_switch_table [1][0] 从3改成6做测试
        [2, 1, 2, 3, 4, 4, 6, 6],
        [5, 1, 5, 3, 4, 5, 6, 6],
        [2, 4, 2, 2, 4, 5, 7, 7],
        [7, 1, 3, 3, 4, 7, 6, 7],
        [0, 6, 2, 6, 6, 5, 6, 7]]

    installRT(p4info_helper, switches, next_switch_table)
    print("*************  install flow tables !!! ")
    time.sleep(5)
    modifyRT(p4info_helper, src_sw=switches[0], next_sw=7, dst_host=host_IPs[1])
    print("*************  install 11111111!!! ")
    time.sleep(5)
    modifyRT(p4info_helper, src_sw=switches[0], next_sw=4, dst_host=host_IPs[1])
    print("*************  install 222222222!!! ")
    # writeRT(p4info_helper, src_sw=switches[1], port=6, dst_host=host_IPs[0])

if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print('耗时： %f ms' % (run_time * 1000))


