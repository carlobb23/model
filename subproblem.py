import gurobipy as gu

class Subproblem:
    def __init__(self, duals_p, duals_td, data, p, iteration, R_p, Entry_p, W, W_min, E_app, E_min):
        itr = iteration + 1
        self.P = p
        self.T = data['T'].dropna().astype(int).unique().tolist()
        self.D = data['D'].dropna().astype(int).unique().tolist()
        self.duals_p = duals_p
        self.duals_td = duals_td
        self.Model = gu.Model("Subproblem")
        self.itr = itr
        self.R_p = R_p
        self.Entry_p = Entry_p
        self.W = W
        self.W_min = W_min  
        self.E_app = E_app  
        self.E_Min = E_min

    def buildModel(self):
        self.genVars()
        self.genCons()
        self.genObj()
        self.Model.update()

    def genVars(self):
        self.e = self.Model.addVars([self.P], self.D, lb=0, vtype=gu.GRB.CONTINUOUS, name="e")
        self.l = self.Model.addVars([self.P], self.D, vtype=gu.GRB.BINARY, name="l")
        self.LOS = self.Model.addVars([self.P], [self.itr], vtype=gu.GRB.INTEGER, name="LOS")
        self.x = self.Model.addVars([self.P], self.T, self.D, [self.itr], vtype=gu.GRB.BINARY, name="x")
        self.y = self.Model.addVars([self.P], self.D, vtype=gu.GRB.BINARY, name="y")
        self.z = self.Model.addVars([self.P], self.T, vtype=gu.GRB.BINARY, name="z")
        self.w = self.Model.addVars([self.P],self. D, vtype=gu.GRB.BINARY, name="w")

    def genCons(self):
        for p in [self.P]:
            self.Model.addLConstr(1 == gu.quicksum(self.l[p, d] for d in self.D))
            self.Model.addLConstr(self.LOS[p, self.itr] == gu.quicksum(self.l[p, d] * d for d in self.D) - self.Entry_p[p] + 1)
            self.Model.addLConstr(1 == gu.quicksum(self.z[p, t] for t in self.T))
            self.Model.addLConstr(1 == gu.quicksum(self.x[p, t, self.Entry_p[p], self.itr] for t in self.T))
            for d in range(2, len(self.D) + 1):
                self.Model.addLConstr(self.w[p, d] <= 1 - gu.quicksum(self.l[p, k] for k in range(2, d)))
            for d in self.D:
                if d < self.Entry_p[p]:
                    self.Model.addLConstr(0 == self.w[p, d])
                elif d == self.Entry_p[p]:
                    self.Model.addLConstr(1 == self.w[p, d])
                self.Model.addLConstr(self.l[p, d] <= gu.quicksum(self.x[p, t, d, self.itr] for t in self.T))
                self.Model.addLConstr(self.l[p, d] <= self.e[p, d])
                self.Model.addLConstr(self.w[p, d] == gu.quicksum(self.x[p, t, d, self.itr] for t in self.T) + self.y[p, d])
                self.Model.addLConstr(self.R_p[p] * self.e[p, d] == gu.quicksum(
                    gu.quicksum(self.x[p, t, j, self.itr] for j in range(1, d + 1)) for t in self.T) + gu.quicksum(
                    self.y[p, j] for j in range(1, d + 1)) * self.E_app)
                if d >= self.Entry_p[p] and d < len(self.D) - self.W + 1:
                    self.Model.addLConstr(gu.quicksum(self.y[p, j] for j in range(d, d + self.W)) <= self.W_min)
                for t in self.T:
                    self.Model.addLConstr(self.x[p, t, d, self.itr] <= self.z[p, t])
        self.Model.update()

    def genObj(self):
        self.Model.setObjective(self.LOS[self.P, self.itr] - gu.quicksum(self.x[self.P, t, d, self.itr] * self.duals_td[t, d] for t in self.T for d in self.D) - self.duals_p[self.P], sense=gu.GRB.MINIMIZE)

    def getOptVals(self, var_name):
        variable = getattr(self, var_name, None)
        if variable is None:
            raise AttributeError(f"Variable '{var_name}' not found in the class.")
        return self.Model.getAttr("X", variable)

    def solModel(self):
        self.Model.Params.OutputFlag = 0
        self.Model.optimize()