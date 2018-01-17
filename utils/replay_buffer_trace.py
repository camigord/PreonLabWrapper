"""
Data structure for implementing experience replay with traces from episodes
"""
from collections import deque
import random
import pickle
import numpy as np

class ReplayBufferTrace(object):

    def __init__(self, buffer_size, trace_length, save_dir, random_seed=123):
        """
        The right side of the deque contains the most recent episodes
        """
        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.count = 0
        self.trace_length = trace_length
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, trajectory):
        if self.count < self.buffer_size:
            self.buffer.append(trajectory)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(trajectory)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        assert (self.trace_length <= len(batch[0])),"Trace can not be longer than episode!"

        sampledTraces = []
        for episode in batch:
            # Get traces from sampled episodes
            point = np.random.randint(0,len(episode) + 1 - self.trace_length)
            sampledTraces.append(episode[point:point+self.trace_length])

        sampledTraces = np.array(sampledTraces)

        return sampledTraces

    def clear(self):
        self.deque.clear()
        self.count = 0

    def save_pickle(self):
        try:
            pickle.dump(obj=self.buffer ,file=open(self.save_dir+'/buffer.p',mode='wb'))
            print("Successfuly saved: RDPG Buffer")
            print("Buffer length saved: " + str(self.count))
        except Exception as e:
            print("Error on saving buffer: " + str(e))

    def load_pickle(self):
        try:
            self.buffer = pickle.load(file=open(self.save_dir+'/buffer.p',mode='rb'))
            self.count = len(self.buffer)
            print("Successfuly loaded: RDPG Buffer")
            print("Buffer length loaded: " + str(self.count))
        except Exception as e:
            print("Could not find old buffer: " + str(e))
