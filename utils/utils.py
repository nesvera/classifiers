class Average():
    
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add_value(self, value):
        self.sum += value
        self.count += 1

    def get_size(self):
        return self.count

    def get_average(self):
        self.avg = float(self.sum)/self.count
        
        return self.avg
