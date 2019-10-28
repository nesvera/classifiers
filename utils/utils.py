import numpy as np

class Average():
    
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 10000000
        self.max = 0

    def add_value(self, value):
        self.sum += value
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def get_size(self):
        return self.count

    def get_average(self):
        self.avg = float(self.sum)/self.count
        
        return self.avg

class SGDR:
    
    def __init__(self, 
                 min_lr=1e-5,
                 max_lr=1e-2,
                 lr_decay=0,
                 epochs_per_cycle=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.epochs_per_cycle = epochs_per_cycle
        self.mult_factor = mult_factor

        self.epoch_since_restart = 0

    def update(self):
        
        cycle_fraction = self.epoch_since_restart/self.epochs_per_cycle
        lr = self.min_lr + 0.5*(self.max_lr-self.min_lr)*(1+np.cos(cycle_fraction*np.pi))

        if self.epoch_since_restart > int(self.epochs_per_cycle):
            self.epoch_since_restart = 0

            self.max_lr = self.max_lr - self.max_lr*self.lr_decay 
            if self.max_lr < self.min_lr:
                self.max_lr = self.min_lr

            self.epochs_per_cycle = self.epochs_per_cycle + self.epochs_per_cycle*self.mult_factor

        else:
            self.epoch_since_restart += 1

        return lr
