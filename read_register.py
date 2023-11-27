#!/usr/bin/env python
# coding=utf-8

import subprocess 

if __name__ == '__main__':
    register_names = ["reg_enq_timestamp", "reg_enq_qdepth", "reg_deq_timedelta", "reg_deq_qdepth", "reg_ingress_global_timestamp", "reg_egress_global_timestamp"]
    ''' # 统计数据量
    path = "./output/h1_with_ca/reg_enq_timestamp.csv"
    with open(path, 'r') as file:
        a = file.readlines()
        print(a[0].count(","))
    '''
    for i in range(8):
        for register_name in register_names:
            p = subprocess.Popen('simple_switch_CLI --thrift-port 909%d'%i,shell=True,stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                    universal_newlines=True) 
            p.stdin.write('register_read %s' % register_name)
            # p.stdin.write('register_read reg_enq_qdepth')
            out, err = p.communicate()  # out 就是 str 类型
            begin_idx = out.find("=") + 2   # 读register总是有前面的一些没用的文字，剔除掉
            out = out[begin_idx: -13]   # out 总是默认会加上 RuntimeCmd: 一共12个字符，我们把最后一行删掉
            
            # with open('output/test_%s.csv'%register_name, 'w') as file:
            #     file.write(out)
            with open('output/h%d_with_ca/%s.csv'%(i+1, register_name), 'w') as file:
                file.write(out)
            # with open('output/h%d_without_ca/%s.csv'%(i+1, register_name), 'w') as file:
            #     file.write(out)
                 
           