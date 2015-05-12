import numpy as np


class AccuracyTable(object):

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
    

class StoppingCriteria(object):
    def __init__(self, k=5):
        self.t = 0
        self.k = k
        self.E_tr = [np.inf]
        self.E_va = [np.inf]
        self.E_opt = np.inf
        
    def append(self, E_tr, E_va):
        self.t += 1
        self.E_tr.append(E_tr)
        self.E_va.append(E_va)
        self.E_opt = min(self.E_opt, E_va)

    @property
    def generalization_loss(self):
        return 100. * (self.E_va[-1]/self.E_opt - 1)

    @property
    def training_progress(self):
        return 1000. * (sum(self.E_tr[-self.k:]) / (self.k * min(self.E_tr[-self.k:])) - 1)

    # stop as soon as the generalization loss exceeds a certain threshold
    def GL(self, alpha):
        return self.generalization_loss > alpha

    # stop as soon as quotient of generalization loss and progress exceeds a certain threshold
    def PQ(self, alpha):
        return self.generalization_loss / self.training_progress > alpha
    
    # stop when the generalization error increased in s successive strips
    def UP(self, s, t=0):
        if t == 0:
            t = self.t
        if t - self.k < 0 or self.E_va[t] <= self.E_va[t - self.k]:
            return False
        if s == 1:
            return True
        return self.UP(s - 1, t - self.k)
