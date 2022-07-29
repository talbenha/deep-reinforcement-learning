import random
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # TODO
        if len(self.memory) >= self.capacity:   # when the memory is full use FIFO
            self.memory[self.position % self.capacity] = Transition(*args)
        else:   # when the memory is not full use append
            self.memory.append(Transition(*args))
        self.position += 1


    def sample(self, batch_size):
        # TODO
        # sample batch from the memory
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)
