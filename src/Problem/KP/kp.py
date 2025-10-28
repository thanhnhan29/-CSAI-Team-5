import numpy as np
import matplotlib.pyplot as plt

class KP:
    def __init__(self, num, capacity, nmax = 100, nmin = 0, seed=42):
        self.num = num
        self.capacity = capacity
        self.weights = []
        self.values = []
        self.nmax = nmax
        self.nmin = nmin
        self.seed = seed
        
    def set_init(self):
        np.random.seed(self.seed)
        self.weights = np.random.randint(self.nmin, self.capacity, size=self.num)
        self.values = np.random.randint(self.nmin, self.nmax, size=self.num)

    
    def solve(self, return_picked = False):
        n = len(self.values)
        dp = [[0]*(self.capacity+1) for _ in range(n+1)]

        for i in range(1, n+1):
            for w in range(self.capacity+1):
                if self.weights[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w],
                                dp[i-1][w-self.weights[i-1]] + self.values[i-1])
                else:
                    dp[i][w] = dp[i-1][w]
        w = self.capacity
        picked = [0]*n
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                picked[i-1] = 1
                w -= self.weights[i-1]
        if return_picked:
            return dp[n][self.capacity], picked
        else:
            return dp[n][self.capacity]    