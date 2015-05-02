import numpy as np

class AccuracyTable():

    def __init__(self, pred=None, obs=None):
        self.table = np.zeros(shape=(3, 3), dtype=float)
        if pred is not None and obs is not None:
            self.count(pred, obs)

    def count(self, pred, obs):
        for i in range(len(pred)):
            self.table[obs[i]][pred[i]] += 1

    @property
    def Q3(self):
        return self.table.trace() / self.table.sum() * 100

    def correlation_coefficient(self, p, n, o, u):
        return (p*n-o*u) / ((p+o)*(p+u)*(n+o)*(n+u))**0.5

    @property
    def Ch(self):
        p = self.table[0][0]
        n = self.table[1][1] + self.table[2][2]
        o = self.table[1][0] + self.table[2][0]
        u = self.table[0][1] + self.table[0][2] + self.table[1][2] + self.table[2][1]
        return self.correlation_coefficient(p, n, o, u)

    @property
    def Ce(self):
        p = self.table[1][1]
        n = self.table[0][0] + self.table[2][2]
        o = self.table[0][1] + self.table[2][1]
        u = self.table[1][0] + self.table[2][0] + self.table[0][2] + self.table[1][2]
        return self.correlation_coefficient(p, n, o, u)

    @property
    def Cc(self):
        p = self.table[2][2]
        n = self.table[0][0] + self.table[1][1]
        o = self.table[0][2] + self.table[1][2]
        u = self.table[1][0] + self.table[2][0] + self.table[0][1] + self.table[2][1]
        return self.correlation_coefficient(p, n, o, u)

    @property
    def C3(self):
        return np.mean((self.Ch, self.Ce, self.Cc))
    

