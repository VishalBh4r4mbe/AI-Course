import time
import numpy as np
'''Basically for benchmarking'''
class Timer:
    def __init__(self):
        self.times = [] #for maintaing the times on the same Timer Object
        self.start()
    def start(self):
        self.initial = time.time()  
    def stop(self):
        self.times.append(time.time() - self.initial)
        return self.times[-1]

    def avg_time(self):
        return np.mean(self.times)
    def sum(self):
        return np.sum(self.times)
    def reset(self):
        self.times = []
        
