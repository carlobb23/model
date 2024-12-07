import gurobipy as gu
import math
import time

class Problem:
    def __init__(self, data, W, W_min, R_p, Entry_p, Max_t, E_app, E_min):
        self.P = data['P'].dropna().astype(int).unique().tolist()
        self.D = data['D'].dropna().astype(int).unique().tolist()
        self.T = data['T'].dropna().astype(int).unique().tolist()
        self.Model = gu.Model("Compact")
        self.W = W
        self.W_min = W_min
        self.R_p = R_p
        self.Entry_p = Entry_p
        self.Max_t = Max_t
        self.E_app = E_app
        self.E_min = E_min

    def buildModel(self):
        self.t0 = time.time()
        self.genVars()
        self.genCons()
        self.genObj()
        self.Model.update()

    def genVars(self):
        self.e = self.Model.addVars(self.P, self.D, lb=0, vtype=gu.GRB.CONTINUOUS, name="e")
        self.l = self.Model.addVars(self.P, self.D, vtype=gu.GRB.BINARY, name="l")
        self.LOS = self.Model.addVars(self.P, vtype=gu.GRB.INTEGER, name="LOS")
        self.x = self.Model.addVars(self.P, self.T, self.D, vtype=gu.GRB.BINARY, name="x")
        self.y = self.Model.addVars(self.P, self.D, vtype=gu.GRB.BINARY, name="y")
        self.z = self.Model.addVars(self.P, self.T, vtype=gu.GRB.BINARY, name="z")
        self.w = self.Model.addVars(self.P, self.D, vtype=gu.GRB.BINARY, name="w")

    def solveStart(self):
        self.Model.Params.MIPGap = 0.9
        self.Model.update()
        self.Model.optimize()

    def genCons(self):
        for t in self.T:
            for d in self.D:
                self.Model.addLConstr(gu.quicksum(self.x[p, t, d] for p in self.P) <= self.Max_t[t, d])
        for p in self.P:
            self.Model.addLConstr(1 == gu.quicksum(self.l[p, d] for d in self.D))
            self.Model.addLConstr(self.LOS[p] == gu.quicksum(self.l[p, d] * d for d in self.D) - self.Entry_p[p] + 1)
            self.Model.addLConstr(1 == gu.quicksum(self.z[p, t] for t in self.T))
            self.Model.addLConstr(1 == gu.quicksum(self.x[p, t, self.Entry_p[p]] for t in self.T))
            for d in range(2, len(self.D) + 1):
                self.Model.addLConstr(self.w[p, d] <= 1 - gu.quicksum(self.l[p, k] for k in range(2, d)))
            for d in self.D:
                if d < self.Entry_p[p]:
                    self.Model.addLConstr(0 == self.w[p, d])
                elif d == self.Entry_p[p]:
                    self.Model.addLConstr(1 == self.w[p, d])
                self.Model.addLConstr(self.l[p, d] <= gu.quicksum(self.x[p, t, d] for t in self.T))
                self.Model.addLConstr(self.l[p, d] <= self.e[p, d])
                self.Model.addLConstr(self.w[p, d] == gu.quicksum(self.x[p, t, d] for t in self.T) + self.y[p, d])
                self.Model.addLConstr(self.R_p[p] * self.e[p, d] == gu.quicksum(
                    gu.quicksum(self.x[p, t, j] for j in range(1, d + 1)) for t in self.T) + gu.quicksum(
                    self.y[p, j] for j in range(1, d + 1)) * self.E_app)
                if d >= self.Entry_p[p] and d < len(self.D) - self.W + 1:
                    self.Model.addLConstr(gu.quicksum(self.y[p, j] for j in range(d, d + self.W)) <= self.W_min)
                for t in self.T:
                    self.Model.addLConstr(self.x[p, t, d] <= self.z[p, t])
        self.Model.update()

    def genObj(self):
        self.Model.setObjective(gu.quicksum(self.LOS[p] for p in self.P), gu.GRB.MINIMIZE)

    def solveModel(self):
        self.Model.Params.MIPGap = 0
        self.Model.optimize()

    def setStart(self, start_dict):
        for key, value in start_dict.items():
            self.x[key].Start = value
        self.model.Params.MIPFocus = 3
        self.model.update()