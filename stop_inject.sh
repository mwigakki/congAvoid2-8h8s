#!/bin/bash
# 需要先sudo su root 再运行此sh

# 如果正在重放，就先杀掉tcpreplay进程
num_tcpreplay=$(ps -au | grep tcpreplay|awk 'END{print NR}')
if [ $num_tcpreplay -gt 1 ]; then
  echo "kill replay1!!!"
  ps -ef | grep tcpreplay | grep -v grep | cut -c 9-15 | xargs kill -9
  echo "kill replay2!!!"
	sleep 1s
fi
# 再杀掉screen
num_screen=$(screen -ls|awk 'NR>=2&&NR<=40{print $1}'|awk 'END{print NR}')
# echo $num_screen
if [ $num_screen -gt 1 ]; then
  echo "kill screens1!!!"
  screen -ls|awk 'NR>=2&&NR<=40{print $1}'|awk '{print "screen -S "$1" -X quit"}'|sh  # 传入命令进screen去关
  echo "kill screens2!!!"
	sleep 1s
fi
