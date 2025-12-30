import subprocess as sp
import psutil
import torch
import re
import os
import logging
from threading import Thread
import time
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Monitor(Thread):
    '''Monitor CPU and GPU usage
    For CPU: RSS and % CPU for the process 
    For GPU: % utilization, % Memory, Total Memory (MiB), Used Memory (MiB)"

    Parameter:
        delay: how long to wait (seconds) between checks
    Returns:
        logs gpu/cpu info
        saves out gpu/cpu csv in pwd
    '''
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.daemon = True # runs in background (takes errors from main process)
        self.date_start = datetime.now().strftime("%Y%m%d%H%M")
        self.delay = delay # Time between calls
        self.gpu = torch.cuda.is_available()
        self.gpu_list = []
        self.process = psutil.Process(os.getpid())
        self.cpu_list = []
        self.start()

    def run(self):
        while not self.stopped:
            cpu_items = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                  self.process.memory_info().rss/(1024**3),
                                  self.process.cpu_percent()]
            logger.info("cpu info: {}".format(cpu_items))
            self.cpu_list.append(cpu_items)
            if self.gpu:
                gpu_mem = sp.check_output(["nvidia-smi", 
                             "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used",
                             "--format=csv"], shell=False, stderr = sp.STDOUT)
                gpu_items = re.split(b', |\n', gpu_mem.strip())[6:]
                self.gpu_list.append(gpu_items)
                logger.info(gpu_items)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        logger.info("Memory output saved as _{}".format(self.date_start))
        cpu_df = pd.DataFrame(self.cpu_list, columns = ["timestamp", "rss", "cpu"] )
        cpu_df.to_csv(f"cpumem_{self.date_start}.csv", index = False)
        if self.gpu:
            gpu_df = pd.DataFrame(self.gpu_list, 
                           columns = ["timestamp", "name", "utilization.gpu [%]", 
                                      "utilization.memory [%]", "memory.total [MiB]", "memory.used [MiB]"]) 
            gpu_df.to_csv(f"gpumem_{self.date_start}.csv", index = False)
        

# example
# monitor = Monitor(10)
# func() # function of interest 
# monitor.stop()