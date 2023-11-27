### 实验说明/实验前期修改/注

**NOTE1** : 本例程为了实现P4不要保存任何pcap文件以及任何log，将`../utils/p4runtime_switch.py` 中77行和79行改成如下：
``` python 
# self.pcap_dump = pcap_dump
self.pcap_dump = False
self.enable_debugger = enable_debugger
# self.log_console = log_console
self.log_console = False
```
这样在运行时节省内存，降低磁盘压力。

**NOTE2** : 本例程为了实现P4RUTIME中流表修改操作，将'../../utils/p4runtime_lib/switch.py' 中注释的部分进行修改

**NOTE3** : 本代码所需环境Python 3.7.13 
安装ubuntu 20默认带python 3.8，ubuntu 18默认带python 3.6
如果是系统自带的python，会使用`/usr/local/lib/python3.8/dist-packages`这个目录。
如果是自己安装的python，会使用`/usr/local/lib/python3.8/site-packages`这个目录。
本实验为了不污染默认环境，采用conda虚拟环境做的，安装的conda 4.5.11
关于环境的其他解释
'''
我将162系统自带的python的google.rpc直接拷到pytorch虚拟环境下，
再将虚拟机里面的p4模块拷到pytorch虚拟环境下，
再安装一个google.protobuf模块，就可以直接运行而不需要再添加上面的额外环境了。 在`~/anaconda3/envs/pytorch/lib/python3.7/site-packages`里
'''
**NOTE4** : TCN层数设置，输入层，隐藏层 1 层，输出层


### 实验运行步骤
1. 打开2个shell，所有运行步骤都在':~/p4/tutorials/exercises/congAvoid2-8h8s'文件夹下

2. 在此文件夹内打开一个shell窗口，输入`make` 运行P4

3. 在第二个 shell 窗口, 
    输入`sudo su`,输入密码，然后输入`source /home/sinet/anaconda3/bin/activate`激活普通用户的conda环境,
    输入 `conda activate pytorch` 使用conda虚拟环境。
    再输入 `python ca.py` 运行拥塞控制python代码
    （自己提前手动结束即ctrl+C ca.py之后，需要手动结束重放流量，步骤如下：    输入`bash stop_inject.sh`）

### 模型训练步骤
目前使用的是TCN1，为简化后的输入矩阵，他的优化器必须选上面那个，也就是`optimizer = optim.Adam(params=[param for model in net.tcn_models for param in model.parameters()], lr=lr, weight_decay=weight_decay)`
TCN2以及其他的学习模型需要选另一个，即`optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)`

输入`sudo su`,
    然后输入`source /home/sinet/anaconda3/bin/activate`激活普通用户的conda环境,
    输入 `conda activate torch` 使用conda虚拟环境。
    再输入 `python model_train_s2h.py` 运行


### 把170上训练好的模块发送到162上，下面的语句在170上执行
    # scp /home/sinet/lt/predict/AttCNN.pt sinet@192.168.199.162:/home/sinet/p4/tutorials/exercises/congAvoid2-8h8s/predict/
   
### P4交换机流量
P4交换机每秒约处理150KB，重放的数据包每个包大小约54B，即每秒约处理2.78K个包
每次实验进行100s，约收到278K个包，在P4交换机中设置每10个包收集一次相关实验数据，则每个数据需要27.8K个空间，



# 文件功能介绍

### basic.p4文件
功能：写了寄存器和计数器，只有普通转发功能
关键字介绍：
（1）ingress_global_timestamp：时间戳（以微秒为单位），当数据包在入口出现时设置。每次开关启动时，时钟都设置为0。可以直接从任一管道（入口和出口）读取此字段，但不应将其写入。++++
（2）egress_global_timestamp：时间戳（以微秒为单位），当数据包开始进行出口处理时设置。时钟与相同ingress_global_timestamp。该字段只能从出口管道中读取，而不能写入。++++
（以下字段只在 出口管道 可以访问）
（1）enq_timestamp：时间戳，以毫秒为单位，设置首次将数据包加入队列的时间。+++
（2）enq_qdepth：首次将数据包排入队列时的队列深度，以数据包数（而不是数据包的总大小）为单位。要在egress里面用++++
（3）deq_timedelta：数据包在队列中花费的时间（以微秒为单位）。+++
（4）deq_qdepth：数据包出队时的队列深度，以数
计数器用于数包，存储包的个数，可以通过P4RUNTIME读取，且读的时候既可以读字节数也可以读包数量

### ca.py文件
功能：首先会安装初始流表，开启重放流量；进入拥塞控制算法循环，目前设置为每 1s 一次（目前算法从预测到计算最佳路由耗时最高为600多ms）

### predict文件夹
功能：存放了TCN1.py代码，以及训练好的模型

### read_register.py 文件
功能：读取寄存器，里面需要选择保存文件的命名方式，（是运行拥塞控制之前的还是之后的）

### modifyTable_test.py 文件夹
功能：实验过程中为了测试如何修改路由表使用的

### stop_inject.sh 和 traffic_inject.sh 文件
功能：开始/停止 重放流量，注意需要root环境

### test.py test.txt
功能：测试临时语句


### 问题排查
- 如果运行`make`后报错：**P4 switch sx did not start correctly.**。多半是sx的端口被占了，切root用户然后`netstat -apn | grep 909x`查一下占用进程的PID，然后`kill -9 [PID]`杀掉就行