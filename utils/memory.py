import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'goal', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)

        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return np.array(batch.state), np.array(batch.goal), np.array(batch.action), np.array(batch.reward), np.array(batch.next_state), np.array(batch.terminal)

    def remove_first(self):
        '''
        Removes and return the first transition in memory and a boolean indicating whether the memory is not empty.
        '''
        if len(self.memory) > 0:
            first_experience = self.memory[0]
            self.memory = self.memory[1:]
            return first_experience, len(self.memory)<1
        else:
            return None, True

    def __len__(self):
        return len(self.memory)
