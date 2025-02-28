import numpy as np
class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.running_mean = 0
        self.running_std = 1
        self.epsilon = epsilon
        self.count = 0
        
    def normalize(self, reward):
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_std = np.sqrt(
            ((self.count - 1) * (self.running_std ** 2) + delta * delta2) / self.count
        )
        
        return (reward - self.running_mean) / (self.running_std + self.epsilon)