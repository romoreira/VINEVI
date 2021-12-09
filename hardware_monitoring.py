import time
import psutil
from csv import writer
import json

record = []

List_Exp = ['time_stamp_begin', 'time_stamp_end','time_spent_on_prediction']
with open('exp_time_spent_on_prediction.csv', 'w') as f:
    writer_object = writer(f)
    writer_object.writerow(List_Exp)
    f.close()

List = ['time_stamp','cpu_util','ctx_switches','cpu_freq','momory_used','memory_swap_used','disk_used','disk_read_count','disk_write_count','temperature']

with open('exp_results.csv', 'w') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(List)
    f_object.close()

#print("Ready to monitor Hardware")

while True:
    
    time_stamp = str(time.time())
    cpu_util = str(psutil.cpu_percent(interval=None))
    ctx_switches = str(psutil.cpu_stats()[0])
    cpu_freq = str(psutil.cpu_freq()[0])
    momory_used = str(psutil.virtual_memory()[2])
    memory_swap_used = str(psutil.swap_memory()[3])
    disk_used = str(psutil.disk_usage('/')[3])
    disk = psutil.disk_io_counters()
    disk_read_count = disk[0]
    disk_write_count = disk[1]
    temperature = dict(psutil.sensors_temperatures().items())
    temperature = temperature.get("cpu_thermal")[0][1]
  
    record.append(time_stamp)
    record.append(cpu_util)
    record.append(ctx_switches)
    record.append(cpu_freq)
    record.append(momory_used)
    record.append(memory_swap_used)
    record.append(disk_used)
    record.append(disk_read_count)
    record.append(disk_write_count)
    record.append(temperature)


    with open('exp_results.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow(record)
        f.close()
    break 
