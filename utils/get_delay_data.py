import subprocess
import time
import re
import xlsxwriter as xw

if __name__ == '__main__':
    i = 100000
    j = 1
    avg = 0
    while i:
        p = subprocess.Popen('simple_switch_CLI --thrift-port 38490',shell=True,stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            universal_newlines=True) 
        # read the int_queue register 1: ingress_timesample 2: egress_timesample
        p.stdin.write('register_read delay 0')

        out,err = p.communicate()

        ingress_delay = re.findall('delay\\[0\\]= (.+?)$', out, re.M)
        p2 = subprocess.Popen('simple_switch_CLI --thrift-port 38490',shell=True,stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            universal_newlines=True) 
        p2.stdin.write('register_read delay 1')

        out,err = p2.communicate()
        egress_delay = re.findall('delay\\[1\\]= (.+?)$', out, re.M)
        result = int(egress_delay[0])-int(ingress_delay[0])
        avg = (avg + result)/j
        result1 = 'result:' + str(result)
        avg1 = 'avg:' + str(avg)
        print(result)
        with open('resultnolcf50knolog.txt','a') as fp:
                fp.write(result1)
                fp.write('\n')
                fp.write(avg1)
                fp.write('\n')
        j = j + 1
        time.sleep(0.3)
        i = i - 1
